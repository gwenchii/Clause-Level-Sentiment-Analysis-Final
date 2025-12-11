from flask import Flask, request, render_template, redirect, url_for
import joblib
import re
import os
from datetime import datetime
import numpy as np

#Discourse Markers
tagalog_discourse_markers = r"\b(?:at|kung|hanggang|hangga’t|bagama’t|nang|o|kaya|pero|dahil\ sa|dahilan\ sa|gawa\ ng|sapagka’t|upang|sakali|noon|sa\ sandali|magbuhat|magmula|bagaman|maliban|bukod|dangan|dahil|yayamang|kapag|pagka|tuwing|matapos|pagkatapos|porke|maski|imbis|sa\ lugar|sa\ halip|miyentras|para|saka|haba|samantala|bago|kundi)\b"
english_discourse_markers = r"\b(?:and|but|or|so|because|although|however|nevertheless|nonetheless|yet|still|despite\ that|in\ spite\ of\ that|even\ so|on\ the\ contrary|on\ the\ other\ hand|otherwise|instead|alternatively|in\ contrast|as\ a\ result|therefore|thus|consequently|hence|so\ that|in\ order\ that|with\ the\ result\ that|because\ of\ this|due\ to\ this|then|next|after\ that|afterwards|since\ then|eventually|finally|in\ the\ end|at\ first|in\ the\ beginning|to\ begin\ with|first\ of\ all|for\ one\ thing|for\ another\ thing|secondly|thirdly|to\ start\ with|in\ conclusion|to\ conclude|to\ sum\ up|in\ short|in\ brief|overall|on\ the\ whole|all\ in\ all|to\ summarize|in\ a\ nutshell|moreover|furthermore|what\ is\ more|in\ addition|besides|also|too|as\ well|in\ the\ same\ way|similarly|likewise|in\ other\ words|that\ is\ to\ say|this\ means\ that|for\ example|for\ instance|such\ as|namely|in\ particular|especially|more\ precisely|to\ illustrate|as\ a\ matter\ of\ fact|actually|in\ fact|indeed|clearly|surely|certainly|obviously|of\ course|naturally|apparently|evidently|no\ doubt|undoubtedly|presumably|frankly|honestly|to\ be\ honest|luckily|fortunately|unfortunately|hopefully|interestingly|surprisingly|ironically)\b"
all_discourse_markers = tagalog_discourse_markers + "|" + english_discourse_markers

#split sentence into clauses
def split_into_clauses(text):
    if not isinstance(text, str):
        return []
    #check dm in the sentence
    has_dm = re.search(all_discourse_markers, text, flags=re.IGNORECASE)
    if has_dm:
        parts = re.split(all_discourse_markers, text, flags=re.IGNORECASE)
    elif ',' in text:
        parts = text.split(',')
    else:
        return [text.strip()]
    clauses = [p.strip() for p in parts if p.strip()]
    return clauses

def extract_discourse_markers(text):
    return re.findall(all_discourse_markers, text, flags=re.IGNORECASE)

#load model
loaded = joblib.load('models/taglish_sentiment_model.pkl', mmap_mode=None)

if isinstance(loaded, dict):
    vectorizer = loaded.get("vectorizer")
    clf = loaded.get("model")
else:
    vectorizer = None
    clf = loaded


app = Flask(__name__)
#temp storage
feedbacks = []

#analyze page
@app.route('/', methods=['GET', 'POST'])
def analyze():
    results = []
    overall_sentiment = None
    percentages = {}
    overall_percent = 0

    if request.method == 'POST':
        user_input = request.form.get('user_input', '').strip()
        if not user_input:
            return render_template('analyze.html', results=[], overall=None, overall_percentage=0, percentages={})

        def preprocess(text):
            return text.lower().strip()

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s*', user_input) if s.strip()]
        sentiment_scores = {c: 0 for c in clf.classes_}

        for sentence in sentences:
            markers_found = extract_discourse_markers(sentence)
            clauses = split_into_clauses(sentence)

            clause_results = []
            sentiment_scores_sentence = {c: 0 for c in clf.classes_}

            for clause in clauses:
                clause_clean = preprocess(clause)

                if vectorizer:
                    X_clause = vectorizer.transform([clause_clean])
                else:

                    X_clause = [clause_clean]

                try:
                    prob_clause = clf.predict_proba(X_clause)[0]
                    pred_clause = clf.classes_[prob_clause.argmax()]
                except Exception as e:

                    prob_clause = np.array([0 for _ in clf.classes_])
                    pred_clause = 'neutral'
                    
                for i, c in enumerate(clf.classes_):
                    sentiment_scores_sentence[c] += prob_clause[i]

                prob_dict = {c.lower(): round(prob_clause[i]*100, 2) for i, c in enumerate(clf.classes_)}
                clause_results.append({
                    'clause': clause,
                    'sentiment': pred_clause,
                    'probabilities': prob_dict
                })

            total_sentence = sum(sentiment_scores_sentence.values())
            percentages_sentence = {k.lower(): round((v/total_sentence)*100, 2) if total_sentence>0 else 0
                                    for k,v in sentiment_scores_sentence.items()}
            overall_sentence = max(percentages_sentence, key=percentages_sentence.get)
            overall_percent_sentence = percentages_sentence[overall_sentence]

            results.append({
                'sentence': sentence,
                'clauses': clause_results,
                'discourse_markers': markers_found,
                'overall': overall_sentence,
                'overall_percentage': overall_percent_sentence,
                'percentages': percentages_sentence
            })

            # Add to overall
            for k in sentiment_scores:
                sentiment_scores[k] += sentiment_scores_sentence[k]

        total = sum(sentiment_scores.values())
        if total > 0:
            percentages = {k.lower(): round((v/total)*100, 2) for k,v in sentiment_scores.items()}
            overall_sentiment = max(percentages, key=percentages.get)
            overall_percent = percentages[overall_sentiment]
        else:
            percentages = {'positive': 0, 'neutral': 0, 'negative': 0}
            overall_sentiment = 'neutral'
            overall_percent = 0

    return render_template(
        'analyze.html',
        results=results,
        overall=overall_sentiment or 'neutral',
        overall_percentage=overall_percent or 0,
        percentages=percentages or {'positive': 0, 'neutral': 0, 'negative': 0}
    )


#feedack page
@app.route("/leave_a_feedback", methods=["GET", "POST"])
def leave_a_feedback():
    global feedbacks
    if request.method == "POST":
        user_input = request.form.get("user_input", "")
        X = vectorizer.transform([user_input]) if vectorizer else [user_input]

        sentiment = clf.predict(X)[0].lower()
        probs = clf.predict_proba(X)[0]
        prob_dict = {c.lower(): round(probs[i]*100, 2) for i, c in enumerate(clf.classes_)}

        feedbacks.insert(0, {
            "text": user_input,
            "sentiment": sentiment,
            "probabilities": prob_dict,
            "timestamp": datetime.now().strftime("%B %d, %Y at %I:%M %p")
        })

        return redirect(url_for("leave_a_feedback"))

    return render_template("feedback.html", feedbacks=feedbacks)

#about page
@app.route('/about_tool')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

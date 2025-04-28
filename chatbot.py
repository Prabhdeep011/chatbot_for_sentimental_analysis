import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

def initialize_sentiment_analyzer():
    sia = SentimentIntensityAnalyzer()
    return sia

def perform_sentiment_analysis(sia, user_input):
    scores = sia.polarity_scores(user_input)
    compound_score = scores['compound']

    if compound_score >= 0.05:
        sentiment = "positive"
    elif compound_score <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return sentiment

def chatbot():
    print("Hello! I am your sentiment analysis chatbot. Type 'exit' to end the conversation.")
    sia = initialize_sentiment_analyzer()

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break

        sentiment = perform_sentiment_analysis(sia, user_input)
        print(f"Bot: The sentiment of your input is {sentiment}.")

if __name__ == "__main__":
    chatbot()

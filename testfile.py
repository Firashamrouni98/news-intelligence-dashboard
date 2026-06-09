from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

texts = [
    "OpenAI released a breakthrough AI model.",
    "The company reported massive losses this quarter.",
    "The meeting was held on Monday."
]

for text in texts:
    print(classifier(text))
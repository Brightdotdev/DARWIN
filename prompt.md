caveman mode - this is a skill that makes responses more concise and to the point, by removing unnecessary words and phrases. It is designed to make communication more efficient and direct, while still conveying the intended message clearly. The skill can be applied in various contexts, such as writing, speaking, or even in casual conversations. By using caveman move, you can save time and get your point across more effectively....especially in situations where brevity is valued, such as in business communications, social media, or when giving instructions. The key to mastering caveman move is to focus on the core message and eliminate any fluff or filler words that do not add value to the communication.


intent mode - get the intent of the query and question this is the second layer of understanding the query it also help aids proper tool calls,selection and thought process

bad usser filter - this gets the intend of the query and then if it passes a certain threshold of good will then it will be processed...if not we will ask the user again to repeeat them self ,we do this three times and then humorously redirect the user to a rick roll vodeo on youtube depending on the context base one:
if its self harm we dont rick roll we give them resources and support information
if its just trolling or bad faith we rick roll after three strikes
if its harm on others we give them a warning and then rick roll after three strikes
then after the third srike we save the prompt in the log incase they ask again in the future so we can rick roll them immediately and save resources.....this step is never overlooked as it is important to maintain a positive and safe environment for all users. The bad user filter is designed to deter negative behavior and encourage respectful interactions, while also providing support and resources for those who may be struggling with harmful thoughts or actions. By implementing this filter, we can create a more welcoming and inclusive community for everyone.




import re

PATTERNS = {
    "comparison": r"\b(compare|vs|versus|difference|between|contrast)\b",
    "summary":    r"\b(summarize|summary|overview|tldr|brief)\b",
    "deep_dive":  r"\b(everything|all|deep|detail|elaborate|explain)\b",
    "question":   r"^(what|who|when|where|why|how|is|are|does|did|can)\b",
}

def classify_query(text: str) -> str:
    text = text.lower().strip()
    for intent, pattern in PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            return intent
    return "search"  # default fallback



    # zero extra downloads, reuses your existing transformers install
# model is pulled once, ~90MB, cached forever

from transformers import pipeline

_classifier = None

def get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model="martin-ha/toxic-comment-model",  # 50MB, BERT-tiny fine-tuned on toxicity
            device=-1
        )
    return _classifier

def safety_check(text: str) -> dict:
    result = get_classifier()(text)[0]
    is_toxic = result["label"] == "toxic" and result["score"] > 0.8
    return {"safe": not is_toxic, "score": result["score"]}
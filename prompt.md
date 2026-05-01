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











    
C:\Users\ADMIN\Documents\studyagent>uv run src/core/Parsing/main.py --file CSB_Pew_Bible_2nd_Printing.pdf --images --caption && uv run src/core/Chunking/main.py --file CSB_Pew_Bible_2nd_Printing.pdf --inspect && uv run src/core/BM25/main.py --file CSB_Pew_Bible_2nd_Printing.pdf --inspect






dels
  ℹ System: 11GB RAM, 11.6GB free
  ✓ All compatible models already cached
  ℹ Searching CSB_Pew_Bible_2nd_Printing.pdf for 'who is david'
  ℹ
==================================================
  ℹ Hybrid Search: 'who is david'
  ℹ RRF k=60 | Semantic weight=1.0 | Keyword weight=1.0
  ℹ Pool size=100 | Top-K=20
  ℹ ==================================================
  ℹ searching semantics... (CSB_Pew_Bible_2nd_Printing.pdf)
  ℹ Using CPU
  ℹ Using USearch index for CSB_Pew_Bible_2nd_Printing.pdf
  ℹ semantic — 100 candidates
  ℹ searching keywords...
  ℹ keyword — 100 candidates
  ℹ fusing results via RRF...
  ✓ done — 20 final results
  ✓ Found 20 results
  ℹ
--- Hybrid Search Results (20 found) ---
  ℹ
[1] Score: 0.9637 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p515)
  ℹ     8 ­Saul ­was fu­ri­ous ­and resented ­this ­song. a “­They cred­it­ed ­tens ­of thou­sands ­to Da­vid,” ­he
com­plained, “­but ­they ­only cred­it­ed ­me ­with thou­sands. ­What ­more ­can ­he ­have ­...
  ℹ     (RRF: 0.009 | KW: 0.964)
  ℹ
[2] Score: 0.9066 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p728)
  ℹ     DAVID’S CAPTURE OF JERUSALEM 4 Da­vid ­and ­all Is­ra­el ­marched ­to Je­ru­sa­lem (­that ­is, Je­bus); ­the
Jeb­u­sites ­who in­hab­it­ed ­the ­land ­were ­there. 5 ­The in­hab­i­tants ­of Je­bus ­sa...
  ℹ     (RRF: 0.030 | Sem: 0.844 | KW: 1.000)
  ℹ
[3] Score: 0.8485 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p758)
  ℹ     ---  [Page 383]  373 1 Chronicles 29  A SUMMARY OF DAVID’S LIFE 26 Da­vid ­son ­of Jes­se ­was ­king ­over ­all
Is­ra­el. 27 ­The ­length ­of ­his ­reign ­over Is­ra­el ­was for­ty ­years; ­he ­reign...
  ℹ     (RRF: 0.015 | Sem: 0.848)
  ℹ
[4] Score: 0.8386 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p1940)
  ℹ     29 “Broth­ers ­and sis­ters, ­I ­can con­fi­dent­ly ­speak ­to ­you ­about ­the pa­tri­arch Da­vid: ­He ­is
­both ­dead ­and bur­ied, ­and ­his ­tomb ­is ­with ­us ­to ­this ­day. 30 ­Since ­he ­was ­...
  ℹ     (RRF: 0.015 | Sem: 0.839)
  ℹ
[5] Score: 0.8383 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p549)
  ℹ     3 ­So ­all ­the el­ders ­of Is­ra­el ­came ­to ­the ­king ­at He­bron. ­King Da­vid ­made ­a cov­enant ­with
­them ­at He­bron ­in ­the ­L­ord’­s pres­ence, ­and ­they anoint­ed Da­vid ­king ­over Is­...
  ℹ     (RRF: 0.019 | Sem: 0.821 | KW: 0.864)
  ℹ
[6] Score: 0.8381 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p529)
  ℹ     DAVID AGAIN SPARES SAUL
  ℹ     (RRF: 0.029 | Sem: 0.838)
  ℹ
[7] Score: 0.8377 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p549)
  ℹ     3 ­So ­all ­the el­ders ­of Is­ra­el ­came ­to ­the ­king ­at He­bron. ­King Da­vid ­made ­a cov­enant ­with
­them ­at He­bron ­in ­the ­L­ord’­s pres­ence, ­and ­they anoint­ed Da­vid ­king ­over Is­...
  ℹ     (RRF: 0.019 | Sem: 0.820 | KW: 0.864)
  ℹ
[8] Score: 0.8365 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p535)
  ℹ     ­the Phi­lis­tine com­mand­ers ­asked, “­What ­are ­these He­brews do­ing ­here? ”  ­Achish an­swered ­the
Phi­lis­tine com­mand­ ers, “­That ­is Da­vid, ser­vant ­of ­King ­Saul ­of Is­ra­el. ­He ­ha...
  ℹ     (RRF: 0.014 | Sem: 0.836)
  ℹ
[9] Score: 0.8335 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p1012)
  ℹ     For the choir director. A psalm of David, when the prophet Nathan came to him after he had gone to Bathsheba. 1
Be gracious to me, God, according to your faithful love; according to your abundant comp...
  ℹ     (RRF: 0.013 | Sem: 0.833)
  ℹ
[10] Score: 0.8334 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p585)
  ℹ     ­These ­are ­the ­last ­words ­of Da­vid: The declaration of David son of Jesse, the declaration of the man
raised on high, h the one anointed by the God of Jacob. This is the most delightful of Israe...
  ℹ     (RRF: 0.013 | Sem: 0.833)
  ℹ
[11] Score: 0.8331 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p1018)
  ℹ     For the choir director: “Do Not Destroy.” A Miktam of David. When he fled before Saul into the cave. 1 Be
gracious to me, God, be gracious to me, for I take refuge in you. I will seek refuge in the sh...
  ℹ     (RRF: 0.012 | Sem: 0.833)
  ℹ
[12] Score: 0.8320 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p730)
  ℹ     ---  [Page 368]  1 Chronicles 12-13 358  31 From half the tribe of Manasseh: 18,000 designated by name to come
and make David king. 32 From the Issacharites, who understood the times and knew what Isr...
  ℹ     (RRF: 0.012 | Sem: 0.832)
  ℹ
[13] Score: 0.8299 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p1416)
  ℹ     17 “­For ­this ­is ­what ­the ­L­ord ­says: Da­vid ­will nev­er ­fail ­to ­have ­a ­man sit­ting ­on ­the
­throne ­of ­the ­house ­of Is­ra­el. 18 ­The Le­vit­i­cal ­priests ­will nev­er ­fail ­to ­ha...
  ℹ     (RRF: 0.012 | Sem: 0.830)
  ℹ
[14] Score: 0.8292 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p591)
  ℹ     DAVID’S DYING INSTRUCTIONS TO SOLOMON
  ℹ     (RRF: 0.011 | Sem: 0.829)
  ℹ
[15] Score: 0.8288 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p730)
  ℹ     8 ­Some Gad­ites de­fect­ed ­to Da­vid ­at ­his strong­hold ­in ­the desert. ­They ­were val­iant war­riors,
­trained ­for bat­tle, ex­pert ­with ­shield ­and ­spear. ­Their fac­es ­were ­like ­the fa...
  ℹ     (RRF: 0.011 | Sem: 0.829)
  ℹ
[16] Score: 0.8286 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p1032)
  ℹ     For the choir director. Of David. To bring remembrance. 1  God, hurry to rescue me. Lord, hurry to help me!  2
Let those who seek to kill me be disgraced and confounded; let those who wish me harm  a6...
  ℹ     (RRF: 0.011 | Sem: 0.829)
  ℹ
[17] Score: 0.8282 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p581)
  ℹ     Dur­ing Da­vid’s ­reign ­there ­was ­a fam­ine ­for ­three suc­ces­sive ­years, ­so Da­vid in­ quired a ­of ­the
­L­ord. ­The ­L­ord an­swered, “­It ­is ­due ­to ­Saul ­and ­to ­his bloody fam­i­ly, b...
  ℹ     (RRF: 0.011 | Sem: 0.828)
  ℹ
[18] Score: 0.8276 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p968)
  ℹ     For the choir director. Of David. The fool says in his heart, “There’s no God.” They are corrupt; they do vile
deeds. There is no one who does good. 2 The Lord looks down from heaven on the human race...
  ℹ     (RRF: 0.011 | Sem: 0.828)
  ℹ
[19] Score: 0.8257 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p1390)
  ℹ     THE RIGHTEOUS BRANCH OF DAVID 5 “Look, the days are coming” ​— ​this is the Lord’s declaration — “when I will
raise up a Righteous Branch for David. He will reign wisely as king and administer justice...
  ℹ     (RRF: 0.010 | Sem: 0.826)
  ℹ
[20] Score: 0.8257 | Source: CSB_Pew_Bible_2nd_Printing.pdf (p1014)
  ℹ     For the choir director: with stringed instruments. A Maskil of David. 1  God, listen to my prayer and do not
hide from my plea for help. 2 Pay attention to me and answer me. I am restless and in turmo...
  ℹ     (RRF: 0.010 | Sem: 0.826)
  ℹ
import wikipedia
import nltk
import yake
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

keyword_extractor=yake.KeywordExtractor()

wiki_first_sentences = wikipedia.summary("Acceleration", sentences=3)
transcript = open("Speech_File.txt", 'r').read()
print(wiki_first_sentences)
print("")
print(transcript)
print('')

deduplication_threshold=0.3
num_of_keywords = 15
custom_kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=deduplication_threshold, top=num_of_keywords, features=None)
keywords_with_frequency = custom_kw_extractor.extract_keywords(wiki_first_sentences)
keywords = []
for item in keywords_with_frequency:
	keywords.append(item[0])

items_mentioned = []
items_not_mentioned = []
for phrase in keywords:
	vectorizer = CountVectorizer().fit_transform([transcript, phrase])
	similarity = cosine_similarity(vectorizer)
	if similarity[0][1] >= 0.1:
		items_mentioned.append(phrase)
	else:
		items_not_mentioned.append(phrase)

print("Items Mentioned: ")
for item in items_mentioned:
	print(item)
print("---------------------")
print("Items Not Mentioned: ")
for item in items_not_mentioned:
	print(item)






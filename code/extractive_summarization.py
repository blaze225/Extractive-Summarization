import os
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import spacy
import math
import scipy
from sklearn.neural_network import BernoulliRBM 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
from rouge import Rouge 

reload(sys)
sys.setdefaultencoding('utf8')

rouge = Rouge()
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
nlp = spacy.load('en')

def sigmoid(x):
  return 1/(1+np.exp(-x))

def getTF_ISF(gDict,eReplacedSents):
	TF_ISF=[]
	for sent in eReplacedSents:
	    LDct={}
	    sent= [stemmer.stem(word) for word in sent.split() if word not in stop_words]
	    # print "=========================================="
	    # print sent
	    # print "=========================================="
	    
	    for word in sent:
	    	if word in stop_words:
	            continue
	        if word not in LDct:
	            LDct[word]=0
	        LDct[word]+=1
	    sum1=0.0
	    for word in sent:
	    	if word in stop_words:
	            continue
	        sum1+=LDct[word]*gDict[word]
	    if sum1==0:
	    	TF_ISF.append(0)
	    else:	
	    	TF_ISF.append(math.log(sum1)/len(sent))

	return TF_ISF
	
def getWordDict(sentences):
	Dict={}
	for sent in sentences:
	    sent=re.sub('[!"#$%&()\'*+,-./:;<=>?[\\]^`{|}~]','',sent)
	    sent= [stemmer.stem(word) for word in sent.split() if word not in stop_words]
	    # print sent
	    for word in sent:
	        if word in stop_words:
	            continue
	        if word not in Dict:
	            Dict[word]=0
	        Dict[word]+=1
	sorted_dictionary = sorted(Dict.iteritems(), key=lambda (k,v): (v,k),reverse=True)[:10]
	sDict={tup[0]:tup[1] for tup in sorted_dictionary}
	return sDict,Dict

def getSentPos(eReplacedSents):
	N=len(eReplacedSents)
	th=0.2*N
	min1=float(th*N)
	max1=min1*2
	return [1 if i==0 or i==N-1 else math.cos((i-min1)*((1/max1)-min1)) for i in range(N)],[1 if i==0 or i==N-1 else 0 for i in range(N)]

def getThematicWordScore(line, NoWords,sDict):
	return float(len([stemmer.stem(word) for word in re.sub(r'[!"#$%&()\'*+,-./:;<=>?[\\]^`{|}~]',' ',line).split() if stemmer.stem(word) in sDict]))/NoWords

def centroidSimilarity(pos,sentences):
	optimal=sentences[pos]
	vOpt=nlp(unicode(optimal))
	cs=[]
	for sent in sentences:
		v=nlp(unicode(sent))
		# cs.append(scipy.spatial.distance.cosine(vOpt.vector,v.vector))
		cs.append(cosine_similarity([vOpt.vector],[v.vector])[0][0])
	return cs

def getvalues(f):
	file = f.read()
	sentences=file.split("\n\n")
	subs = sentences[3].split('\n')
	subDict={line.split(":")[0]:line.split(":")[1] for line in subs}
	
	abstract_summary = " ".join(sentences[2].split('\n'))
	# abstract_summary = " ".join([subDict[word] if word in subDict else word for word in abstract_summary.split()])

	sentences = sentences[1].split('\n')
	sentences=[line.decode('utf-8') for line in sentences]
	temp_sentences=[]
	labels=[]
	for line in sentences:
		l=line.decode('utf-8').split('\t\t\t')
		temp_sentences.append(l[0])
		labels.append(l[1])
	sentences=temp_sentences
	# sentences = [line.decode('utf-8').split('\t\t\t')[0] for line in sentences]

	# labels = [line.decode('utf-8').split('\t\t\t')[1] for line in sentences]
	eReplacedSents=[(" ").join([subDict[word] if word in subDict else word for word in line.split()]) for line in sentences]
	
	ProperNouns=[len([word.pos_ for word in nlp(unicode(line))  if word.pos_=="PROPN"]) for line in eReplacedSents]
	NamedEntities=[len([word.ent_type_ for word in nlp(unicode(line))  if word.ent_type_!=""]) for line in eReplacedSents]
	

	
	eReplacedSents=[re.sub(u'[!"#$%&()\'*+,-./:;<=>?[\\]^`{|}~]',' ',line) for line in eReplacedSents]
	# print 
	# for item in eReplacedSents:
	# 	print item
	# print "\n"
	sDict,gDict=getWordDict(eReplacedSents)
	NoWords=sum(gDict.values())

	ThematicWordScore=[(float(len([stemmer.stem(word) for word in line.split() if stemmer.stem(word) in sDict]))/len(line.split())) if len(line.split())>0 else 0 for line in eReplacedSents]
	
	sentencePosition, sentencePositionPara=getSentPos(eReplacedSents)
	sentenceLength=[0 if len(line.split()) < 3 else len(line.split()) for line in eReplacedSents]
	numerals=[float(len([word.pos_ for word in nlp(unicode(line))  if word.pos_=="NUM"]))/len(line.split()) if len(line.split())>0 else 0 for line in eReplacedSents]
	TS_ISF=getTF_ISF(gDict,eReplacedSents)

	pos=np.argmax(TS_ISF)
	# print "pos",pos,"len",len(eReplacedSents),"tsisf",len(TS_ISF)
	cs=centroidSimilarity(pos,sentences)

	FeatureVectors=[
		[f1,f2,f3,f4,f5,f6,f7,f8,f9]
		for f1,f2,f3,f4,f5,f6,f7,f8,f9 in zip(ThematicWordScore,sentencePosition,sentenceLength,sentencePositionPara,ProperNouns,numerals,NamedEntities,TS_ISF,cs)
		]	

	return FeatureVectors,sentences,labels,abstract_summary

def generateSummary(scores, sentences, labels, file):
	summary=[]
	ref_summary=[]
	summary_limit=len(sentences)/3
	for i in range(summary_limit):
		index = np.argmax(scores)
		summary.append((sentences[index],index, labels[index]))
		scores[index]=-999999999
	summary.sort(key=lambda x:x[1])
	ref_summary=[s[0] for s in summary if s[2]=='1' or s[2]=='2']
	# ref_summary=[s[0] for s in summary if s[2]=='1']
	summary= [(s[0],s[2]) for s in summary]
	## write to file ##
	# f= open('../summaries/'+file,'w')
	# for s in summary:
	# 	f.write(s[0].encode('utf-8')+'\t\t\t'+s[1].encode('utf-8')+'\n')
	return summary, ref_summary

def getScores(labels, summary):
	# total = sum([1 for l in labels if l=='1'])
	total = labels.count('1') + labels.count('2')/float(2)
	summary_size=len(summary)
	summary_total = sum([1 for s in summary if s[1]=='1' ]) + sum([0.5 for s in summary if s[1]=='2' ])
	precision = summary_total/float(summary_size)
	recall = summary_total/float(total)
	if precision==0 and recall==0:
		FScore=0
	else:
		FScore = 2*recall*precision/float(recall+precision)
	return precision,recall,FScore

def getRougeScores(summary, abstract_summary):
	rouge = Rouge()
	scores = rouge.get_scores(summary, abstract_summary)
	return socres

def get_sentences(dir_path='../cnn_test/'):
	files=os.listdir(dir_path)
	# files=['4092a37f920d027d0f8a8c2b50e0af3d3bf5dbea.summary']
	summary=[]
	scores={'precision':0,'recall':0, 'FScore':0}
	files_ignored=0
	summaries=[]
	ref_summaries=[]
	for i,file in enumerate(files):
		print str(i+1)+' -> '+file
		f = open(dir_path+file,'r')
		FeatureVectors,sentences,labels,abstract_summary=getvalues(f)
		if len(sentences)<50:
			files_ignored+=1
			continue
		X=np.array(FeatureVectors)
		model=BernoulliRBM(n_components=9,learning_rate=0.1)
		model.fit(X)
		rbm_scores=model.score_samples(X)

		summary,ref_summary=generateSummary(rbm_scores,sentences,labels,file)
		precision,recall,FScore=getScores(labels,summary)
		print "Precision:",precision," Recall:",recall, " FScore:",FScore
		scores['precision']+=precision
		scores['recall']+=recall
		scores['FScore']+=FScore

		summaries.append(" ".join([s[0] for s in summary]))
		# summaries.append(" ".join([s[0] for s in summary if s[1]=='1']))
		ref_summaries.append(" ".join(ref_summary))

	scores={k: scores[k]/float(len(files)-files_ignored) for k in scores}
	print "files tested: ",len(files)-files_ignored
	print scores
	print rouge.get_scores(summaries, ref_summaries, avg=True)

# get_sentences('../neuralsum/train/cnn_training/')
# get_sentences('../neuralsum/valid/cnn_validation/')
get_sentences()


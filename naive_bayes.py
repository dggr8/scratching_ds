#!/usr/bin/env python

import math
import random
def tokenize(message):
	message = message.lower()
	all_words = re.findall("[a-z0-9']+",message)
	return set(all_words)

def count_words(training_set):
	"""training set consists of pairs (message,is_spam)"""
	counts = defaultdict(lambda:[0,0])
	for message,is_spam in training_set:
		for word in tokenize(message):
			counts[word][0 if is_spam else 1] += 1

	print "Number of words  = ",len(counts.keys())
	return counts

def word_probabilities(counts,total_spams,total_non_spams,k=0.5):
	"""turn the word_counts into a list of triplets
	w, p(w|spam) and p(w| -spam)"""
	return [(w,
			(spam+k)/(total_spams + 2 * k),
			(non_spam+k)/(total_non_spams + 2 * k))
			for w, (spam,non_spam) in counts.iteritems()]

def spam_probability(word_probs,message):
	message_words = tokenize(message)
	log_prob_if_spam = log_prob_if_not_spam = 0.0

	#iterate through each word in our vocabulary
	for word, prob_if_spam , prob_if_not_spam in word_probs:
		
		#if "word" appears in the message,
		#add the log probability of seeing it
		if word in message_words:
			log_prob_if_spam += math.log(prob_if_spam)
			log_prob_if_not_spam += math.log(prob_if_not_spam)

		#if "word" doesn't appear in the message
		#add the log probability of _not_ seeing it
		#which is log(1-probability of seeing it)
		else:
			log_prob_if_spam += math.log(1.0 - prob_if_spam)
			log_prob_if_not_spam += math.log(1.0 - prob_if_not_spam)

	prob_if_spam = math.exp(log_prob_if_spam)
	prob_if_not_spam = math.exp(log_prob_if_not_spam)
	return prob_if_spam / (prob_if_spam +  prob_if_not_spam)

class NaiveBayesClassifier:

	def __init__(self,k=0.5):
		self.k = k
		self.word_probs = []

	def train(self,training_set):

		num_spams = len([is_spam
						for message,is_spam in training_set
						if is_spam])
		num_non_spams = len(training_set) - num_spams

		# run training data through our "pipeline"
		word_counts = count_words(training_set)
		self.word_probs = word_probabilities(word_counts,
											num_spams,
											num_non_spams,
											self.k)

	def classify(self,message):
		return spam_probability(self.word_probs,message)

import glob,re
from collections import Counter
from collections import defaultdict

def split_data(data,prob):
	"""split data into fractions [prob, 1-prob]"""
	results = [],[]
	for row in data:
		results[0 if random.random()< prob else 1].append(row)
	return results

#give path here
path = r"/home/qwerty/coding/scratching_ds/spamassassin/*/*"
data = []

for fn in glob.glob(path):
	is_spam = "ham" not in fn

	with open(fn,'r') as file:
		for line in file:
			if line.startswith("Subject:"):
				#remove the leading "Subject: " and keep what's left
				subject = re.sub(r"^Subject: ","",line).strip()
				data.append((subject,is_spam))

random.seed(0)
train_data,test_data = split_data(data,0.75)

classifier = NaiveBayesClassifier()
classifier.train(train_data)

classified = [(subject,is_spam,classifier.classify(subject))
			for subject,is_spam in test_data]

counts = Counter((is_spam,spam_probability>0.5)
				for _,is_spam,spam_probability in classified)

print counts

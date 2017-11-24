from scipy.stats import entropy

def show_histogram(hist):
	for i in range(len(hist)):
		print("{} : {}".format(i, hist[i]))
	print("entropy : {}".format(entropy(hist)))

def pdf(hist):
	total = max(sum(hist), 1) #avoid division by 0
	distribution = [e/total for e in hist]
	return distribution
	
categories = [0] * 100

while True:
	show_histogram(pdf(categories))
	try:
		num = int(input("next number: "))
	except:
		break
	categories[num] += 1


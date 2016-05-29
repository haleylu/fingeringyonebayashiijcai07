# 2016-5-28
# yimenglu

# initiation codes
import scipy, math, scipy.integrate, sys, random, threading

DEBUG = False
# since we don't need to write a log right now...
# f_trace = open("last_training.log", "w");

states = ('1st', '2nd', '3rd', '4th', '5th')

# Y AXIS
#
# ^                        |<---------164 millimetres------>|
# |                        |                                |
# |       Octave N         |          Octave N+1            |   Octave N+2
# |   .---------^--------. .--------------^-----------------. .------^-----.
# |                        
# |   +--+--+--+--+-+--+--+--+--+--+--+--+--+--+--+--+-+--+--+--+--+--+--+--+   ---
# |   |  |  |  |  | |  |  |  |  |  |  |  |  |  |  |  | |  |  |  |  |  |  |  |    ^
# |   |  |  |  |  | |  |  |  |  |  |  |  |  |  |  |  | |  |  |  |  |  |  |  |    |
# |   |  |  |  |  | |  |  |  |  |  |  |  |  |  |  |  | |  |  |  |  |  |  |  |    |
# |   |  |  |  |  | |  |  |  |  |  |  |  |  |  |  |  | |  |  |  |  |  |  |  |    |
# |   |  |#F|  |#G| |#A|  |  |#C|  |#D|  |  |#F|  |#G| |#A|  |  |#C|  |#D|  | 144 millimetres
# |   |  |__|  |__| |__|  |  |__|  |__|  |  |__|  |__| |__|  |  |__|  |__|  |    |
# |   |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
# |   | F  | G  | A  | B  | C  | D  | E  | F  | G  | A  | B  | C  | D  | E  |    v
# |   |____|____|____|____|____|____|____|____|____|____|____|____|____|____|   ---
# |
# +-----------------------------------------------------------------------------------------> X AXIS
BoundingBoxes0 = { # Format: (Note Name, Octave) = [ List of Bounding Boxes ]
	("F" ,3):  [(0,0,23,50), (0,50,12,144)], #是该键在平面上所占的所有区域
	("#F",3):  [(12,50,26,144)],
	("G" ,3):  [(23,0,47,50),(26,50,39,144)],
	("#G",3):  [(39,50,53,144)],
	("A" ,3):  [(47,0,70,50),(53,50,66,144)],
	("#A",3):  [(66,50,80,144)],
	("B" ,3):  [(70,0,94,50),(80,50,94,144)],
	("C" ,4):  [(94,0,117,50),(94,50,107,144)],
	("#C",4):  [(107,50,122,144)],
	("D", 4):  [(117,0,141,50),(122,50,136,144)],
	("#D",4):  [(136,50,150,144)],
	("E", 4):  [(141,0,164,50),(150,50,164,144)]
}

# Expand to more octaves.
BoundingBoxes = dict()
def ComputeBoundingBoxes(): #计算各个键的占有区域
	global BoundingBoxes, BoundingBoxes0
	for o in range(-3, 4):
		delta_x = 164*o
		for k, v in BoundingBoxes0.items():
			k1 = (k[0], k[1]+o); v1 = []
			for bb in v:
				v1.append((bb[0]+delta_x, bb[1], bb[2]+delta_x, bb[3]))
			BoundingBoxes[k1] = v1
ComputeBoundingBoxes()

##给一个手指和琴键，返回手指中心的落点
NoteNameToIdx = {"F":0, "#F":1, "G":2, "#G":3, "A":4, "#A":5, "B":6, "C":7, "#C":8, "D":9, "#D":10, "E":11}
Note3X        = [12,    19,     35,    46,     59,    73,     82,    -58,   -50,    -35,   -21,     -11   ]#x_middle of a key
NoteIsWhite   = [True,  False,  True,  False,  True,  False,  True,  True,  False,  True,  False,   True  ] 
FingerToIdx   = {'1st':0, '2nd':1, '3rd':2, '4th':3, '5th':4}
# 对手指放在键上时的Y坐标的猜测初始值，后面会进行迭代优化.前五个是1-5指放在白键上的Y坐标，后五个则是黑键
g_Y = [25, 30, 35, 30, 25, 70, 75, 80, 75, 70]
# 在本模型的假设中，x仅仅取决于键位，Y则取决于用哪一个手指和黑白键。因此Y值需要调整，x则不用
def GetContactPoint(no, finger): 
	global NoteNameToIdx, g_Y
	idx = NoteNameToIdx[no[0]]
	octave = no[1]
	x = (octave - 3) * 164 + Note3X[idx]
	is_white = NoteIsWhite[idx]
	fidx = FingerToIdx[finger]
	y0_w = g_Y[0:5]
	y0_b = g_Y[5:10]
	if is_white: y = y0_w[fidx]
	else:        y = y0_b[fidx]
	return (x, y)

# 一些计算用函数
def Gaussian2D_PDF(x, y, x0, y0, sigma_x_sq, sigma_y_sq):
	xx = (x - x0) * (x - x0); yy = (y - y0) * (y - y0)
	return 1.0 / 2 / 3.1415926 / math.sqrt(sigma_x_sq * sigma_y_sq) \
		* math.exp(-0.5 * (xx / sigma_x_sq + yy / sigma_y_sq))

# Refer to these pages for how this works
#
# Computing the error function with a computer (when x is not too large)
# http://math.stackexchange.com/a/996598/228339
#
# When x is large, use another asymptotic approximation:
# http://mathworld.wolfram.com/Erf.html (See Formula 18~20)
def erf(x):
	if abs(x) < 20: 
		a = 1.0; b = 1.5;
		ret = 1.0;
		term = 1.0 * a / b * x * x;
		s = 1
		while True:
			ret = ret + term
			term = term * ((a + s) / (b + s) / (s + 1)) * x * x
			if math.isinf(term):
				print(x, a, b, s)
				assert False
			if abs(term) < 1e-8:
				break
			s = s + 1
		ret = ret * 2 * x / math.sqrt(3.1415926) * math.exp(-x*x)
		return ret
	else:
		term = 1.0 / x;
		ret = 0.0;
		s = 1
		while s < 10:
			ret = ret + term;
			term = term / 2.0 * (s*2-1) / x / x * (-1)
			if abs(term) < 1e-8 or abs(term) > 100: break
			s = s + 1
		ret = 1.0 - math.exp(-x*x) / math.sqrt(3.1415926) * ret
		return ret

def pnorm(x, sigma_x_sq):
	return 0.5 + 0.5 * erf(x / math.sqrt(2 * sigma_x_sq))

def Gaussian2DProbMass(tact, x0, y0, sigma_x_sq, sigma_y_sq, bbs):
	ret = 0.0;
	for bb in bbs:
		# Compute X
		lwr_x = bb[0] - tact[0] - x0
		upr_x = bb[2] - tact[0] - x0
		probmass_x = pnorm(upr_x, sigma_x_sq) - pnorm(lwr_x, sigma_x_sq)
		# Y
		lwr_y = bb[1] - tact[1] - y0
		upr_y = bb[3] - tact[1] - y0
		probmass_y = pnorm(upr_y, sigma_y_sq) - pnorm(lwr_y, sigma_y_sq)

		ret = ret + probmass_x * probmass_y # 联合CDF == 边缘CDF乘积
	return ret

# If zero then return a very small epsilon.
prob_cache = dict()
def EmissionProbability_safe(state_begin, state_end, note_begin, note_end):
	global prob_cache;
	key = (state_begin, state_end, note_begin, note_end);
	if False: pass
	else: 
		ret = EmissionProbability(state_begin, state_end, note_begin, note_end)
		prob_cache[key] = ret
	EPSILON = 1e-30;
	if ret > EPSILON: return ret
	else: return EPSILON

# Compute Emission Probability Mass # 输出概率
# state_{begin,end}: ['1st', '2nd', '3rd', '4th', '5th']
# note_{begin,end}:  ([#][ABCDEFG], [012345678])
# 计算从一个手指、音符对到另一个手音符对转化的概率
def EmissionProbability(state_begin, state_end, note_begin, note_end):#反正仅有一行会被执行
	global prob_cache;
	end_bbs = BoundingBoxes[note_end];# 一个区域（4个值）
	start_tact = GetContactPoint(note_begin, state_begin) # 一个向量（两个值）
	if state_begin == '1st':
		if state_end == '1st': 
			return Gaussian2DProbMass(start_tact, 0, 0, 5, 30, end_bbs)
		elif state_end == '2nd':
			return 0.81 * Gaussian2DProbMass(start_tact,  42, 25, 900, 30, end_bbs) + \
			       0.19 * Gaussian2DProbMass(start_tact, -23, 25, 400, 30, end_bbs)
		elif state_end == '3rd':
			return 0.89 * Gaussian2DProbMass(start_tact,  50, 30, 900, 30, end_bbs) + \
			       0.11 * Gaussian2DProbMass(start_tact, -16, 30, 100, 30, end_bbs)
		elif state_end == '4th':
			return 0.91 * Gaussian2DProbMass(start_tact,  85, 25, 900, 30, end_bbs) + \
			       0.09 * Gaussian2DProbMass(start_tact, -16, 25, 100, 30, end_bbs)
		elif state_end == '5th':
			return 0.95 * Gaussian2DProbMass(start_tact, 110,  0, 900, 30, end_bbs) + \
			       0.05 * Gaussian2DProbMass(start_tact, -21,  0, 400, 30, end_bbs)
		else: assert(False)
	elif state_begin == '2nd':
		if state_end == '1st': 
			return 0.81 * Gaussian2DProbMass(start_tact, -42, -25, 900, 30, end_bbs) + \
			       0.19 * Gaussian2DProbMass(start_tact,  23, -25, 400, 30, end_bbs)
		elif state_end == '2nd':
			return Gaussian2DProbMass(start_tact, 0, 0, 5, 30, end_bbs)
		elif state_end == '3rd':
			return Gaussian2DProbMass(start_tact, 23, 10, 180, 30, end_bbs)
		elif state_end == '4th':
			return Gaussian2DProbMass(start_tact, 50, 0, 200, 30, end_bbs)
		elif state_end == '5th':
			return Gaussian2DProbMass(start_tact, 82, -25, 200, 30, end_bbs)
		else: assert(False)
	elif state_begin == '3rd':
		if state_end == '1st':
			return 0.89 * Gaussian2DProbMass(start_tact, -50, -30, 900, 30, end_bbs) + \
			       0.11 * Gaussian2DProbMass(start_tact,  16, -30, 100, 30, end_bbs)
		elif state_end == '2nd':
			return Gaussian2DProbMass(start_tact, -23, 10, 180, 30, end_bbs)
		elif state_end == '3rd':
			return Gaussian2DProbMass(start_tact,   0,  0, 5,   30, end_bbs)
		elif state_end == '4th':
			return Gaussian2DProbMass(start_tact,  18, -10, 190, 30, end_bbs)
		elif state_end == '5th':
			return Gaussian2DProbMass(start_tact, 57, -25, 250, 30, end_bbs)
		else: assert(False)
	elif state_begin == '4th':
		if state_end == '1st':
			return 0.91 * Gaussian2DProbMass(start_tact, -85, -25, 900, 30, end_bbs) + \
			       0.09 * Gaussian2DProbMass(start_tact, 16, -25,  100, 30, end_bbs)
		elif state_end == '2nd':
			return Gaussian2DProbMass(start_tact, -50, 0, 200, 30, end_bbs)
		elif state_end == '3rd':
			return Gaussian2DProbMass(start_tact, -18, 10, 190, 30, end_bbs)
		elif state_end == '4th':
			return Gaussian2DProbMass(start_tact, 0, 0, 5, 30, end_bbs)
		elif state_end == '5th':
			return Gaussian2DProbMass(start_tact, 20, -20, 200, 30, end_bbs);
		else: assert(False)
	elif state_begin == '5th':
		if state_end == '1st':
			return 0.95 * Gaussian2DProbMass(start_tact, -110, 0, 900, 30, end_bbs) + \
			       0.05 * Gaussian2DProbMass(start_tact,  21,  0, 400, 30, end_bbs)
		elif state_end == '2nd':
			return Gaussian2DProbMass(start_tact, -82, 25, 200, 30, end_bbs);
		elif state_end == '3rd':
			return Gaussian2DProbMass(start_tact, -57, 25, 250, 30, end_bbs)
		elif state_end == '4th':
			return Gaussian2DProbMass(start_tact, -20, 20, 200, 30, end_bbs)
		elif state_end == '5th':
			return Gaussian2DProbMass(start_tact, 0, 0, 5, 30, end_bbs)
		else: assert(False)
	else: assert(False)



def GetPathCost(obs, path, start_p, trans_p, emission_probability):#一步一步计算这条路上的cost
	f_prev = None; notes_prev = None; sum_p = 0;
	for t in range(0, len(obs)):
		f_curr = path[t]
		notes_curr = obs[t]
		ret, _, _, _ = ComputeAllTransitionCost(f_prev, f_curr, notes_prev, notes_curr, start_p, trans_p,
			emission_probability);
		sum_p += ret
		f_prev = f_curr
		notes_prev = notes_curr
	return sum_p    
# Called for each new note (or note group)
# If f_prev is None or notes_prev is None, then compute the probability
# for the first fingering decision
# 计算path之上，进行一次转换的cost
def ComputeAllTransitionCost(f_prev, f_curr, notes_prev, notes_curr, start_p, trans_p, emission_probability):
	ret = 0; tran_p = 0; emi_p = 0; vert_p = 0
	# Part 1: Transition Probability
	if f_prev is None or notes_prev is None:
		for f1 in f_curr:
			tran_p += math.log(start_p[f1])
		ret += tran_p;
	else:
		for nidx1, f1 in enumerate(f_curr):
			for nidx0, f0 in enumerate(f_prev):
				tran_p += math.log(trans_p[f0][f1])# 隐马尔可夫转换概率
				emi_p += math.log(emission_probability(f0, f1, notes_prev[nidx0], notes_curr[nidx1]))# Emission转换概率
		ret = ret + tran_p
		ret = ret + emi_p 
	
	# Part 2: Inside-a-chord cost
	if len(f_curr) > 1:
		for i in range(0, len(f_curr)-1):
			vert_p += math.log(trans_p[f_curr[i]][f_curr[i+1]])
	ret = ret + vert_p
	diff = ret - vert_p - emi_p - tran_p
	if abs(diff) > 1e-5: assert(False)
	return ret, tran_p, emi_p, vert_p

def do_Permutation(elts, num, stack, last_idx):
	if len(stack) == num:
		yield tuple(stack)
	else:
		for i in range(last_idx, len(elts)):
			e = elts[i]
			if not e in stack:
				for x in do_Permutation(elts, num, stack[:]+[e], i+1):
					yield x
def Permutation(elts, num):
	for x in do_Permutation(elts, num, [], 0):
		yield x

# viterbi, 最后给出path和cost
def viterbiPoly(obs, states, start_p, trans_p, emission_probability):
	Vprev = {} # Key: tuple, states at the previous Time Step
	           # Value: probability and note played
	
	Vcurr = {}

	path = {}
	path_prev = {}

	for y in states:
		Vprev[tuple([y])] = (0, (None))
		path[tuple([y])]  = [([y])]
	
	for t in range(0, len(obs)):
		if DEBUG: print("t=%d" % t)
		V = {}
		curr_notes = obs[t]
		if type(curr_notes) == tuple:
			curr_notes = [curr_notes]

		# All possible fingerings of this "unit"
		# (Assuming notes in a chord are in ascending order)
		num_notes = len(curr_notes)

		k0 = sorted(Vprev.keys(), key=lambda x : Vprev[x][0], reverse=True)[0]
		if DEBUG: print(k0, Vprev[k0][1])

		for fingers in Permutation(states, num_notes):
			max_lg_prob = -1e20
			for fingers_prev, prob_and_notes_prev in Vprev.items():
				delta_lg_prob = 0

				lg_prob_prev  = prob_and_notes_prev[0]
				notes_prev = prob_and_notes_prev[1]

				delta_lg_prob, _, _, _ = ComputeAllTransitionCost(fingers_prev, fingers,
					notes_prev, curr_notes, start_p, trans_p, emission_probability);

				if lg_prob_prev + delta_lg_prob > max_lg_prob:
					Vcurr[fingers] = (lg_prob_prev + delta_lg_prob, tuple(curr_notes))
					if t > 0: path[fingers] = path_prev[fingers_prev]
					else: path[fingers] = []
					max_lg_prob = lg_prob_prev + delta_lg_prob
					if DEBUG: print("%s %s ---> %s %s = %g" % (
						str(fingers_prev), str(notes_prev),
						str(fingers),      str(curr_notes),
						lg_prob_prev + delta_lg_prob))

		Vprev = Vcurr; Vcurr = {}
		for k, v in path.items():
			path_prev[k] = v + [k]
		path = {}
	k0 = sorted(Vprev.keys(), key=lambda x : Vprev[x][0], reverse=True)[0]
	the_path = path_prev[k0]
	return the_path, Vprev[k0][0]


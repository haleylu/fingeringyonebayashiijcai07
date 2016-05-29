from initiation import *

# training data
start_p1 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations1 = [
	# Measure 1
    [("C",3)],[("D",3)],[("E",3)],[("F",3)],[("D",3)],[("E",3)],[("C",3)], \
	[("G",3)],[("C",4)],[("B",3)],[("C",4)],[("B",3)],[("C",4)],
	# Measure 2 
	[("D",4)],[("G",3)],[("A",3)],[("B",3)],[("C",4)],[("A",3)],[("B",3)],[("G",3)],[("D",4)],[("G",4)] ]
y0_1 = [ ["1st"], ["2nd"], ["3rd"], ["4th"], ["2nd"], ["3rd"], ["1st"], ["2nd"], ["4th"], ["3rd"], ["4th"], 
         ["3rd"], ["4th"], \
		 ["5th"], ["1st"], ["2nd"], ["3rd"], ["4th"], ["2nd"], ["3rd"], ["1st"], ["3rd"], ["5th"] ]

start_p2 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.21} # Somewhere in the middle so we can set the probabilites
observations2 = [
	[("A",4)], [("G",4)], [("F",4)], [("E",4)], [("D",4)], [("C",4)], [("E",4)], [("D",4)], [("F",4)]
]
y0_2 = [ ["5th"], ["4th"], ["3rd"], ["2nd"], ["1st"], ["2nd"], ["4th"], ["3rd"], ["5th"] ]

start_p3 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations3 = [ [("G",3)], [("#D",4)], [("D",4)], [("#D",4)], [("#A",3)], [("C",4)], [("#D",4)], [("#G",4)] ]
fingers3_me = [ ["1st"],   ["3rd"],    ["1st"],   ["4th"] ]
fingers3_x  = [ ["1st"],   ["4th"],    ["3rd"],   ["4th"] ]
y0_3 = [ ["1st"], ["4th"], ["3rd"], ["4th"], ["2nd"], ["1st"], ["3rd"], ["5th"] ]

start_p4 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations4 = [
	[("C",4)], [("#A",3)],[("#G",3)],[("G",3)],[("#F",3)],[("#D",4)],[("D",4)],[("C",4)],
	[("#A",3)],[("A",3)],[("G",3)],[("F",3)] ]
fingers4_me = [ ["5th"],  ["3rd"],  ["2nd"],  ["1st"] ]
fingers4_x  = [ ["5th"],  ["4th"],  ["3rd"],  ["1st"] ]
y0_4 = [ ['5th'], ['4th'], ['3rd'], ['1st'], ['2nd'], ['5th'], ['4th'], ['3rd'], ['2nd'], ['1st'],
         ['3rd'], ['2nd'] ]

start_p5 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations5 = [
	[("G",3)], [("G",3)], [("A",3)], [("B",3)]
]
y0_5 = [ ["2nd"], ["2nd"], ["3rd"], ["4th"] ]

start_p6 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations6 = [
	[("#G",3)], [("F",4)], [("E",4)], [("D",4)], [("C",4)], [("D",4)], [("C",4)],
	[("B",3)],  [("A",3)],
	[("A",3)],  [("A",4)], [("G",4)],  [("F",4)], [("E",4)], [("G",4)], [("F",4)], [("A",4)], [("G",4)]
]
y0_6 = [ ["1st"], ["5th"], ["4th"], ["3rd"], ["2nd"], ["4th"], ["3rd"],
         ["2nd"], ["1st"], ["1st"], ["5th"], ["4th"], ["3rd"], ["2nd"], ["4th"], ["3rd"], ["5th"], ["4th"] ]

start_p7 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations7 = [
	[("A",3)], [("D",3)], [("C",4)], [("B",3)], [("C",4)], [("C",4)], [("D",4)],
	[("B",3)], [("A",3)], [("G",3)], [("#F",3)], [("E",3)], [("G",3)], [("#F",3)],
	[("A",3)], [("G",3)], [("B",3)], [("A",3)], [("C",4)], [("B",3)], [("D",4)],
	[("C",4)], [("E",4)], [("D",4)], [("B",3)], [("C",4)]
]
y0_7 = [ ["4th"], ["1st"], ["4th"], ["3rd"], ["4th"], ["4th"], ["5th"],
         ["3rd"], ["2nd"], ["1st"], ["3rd"], ["2nd"], ["3rd"], ["2nd"], ["3rd"],
		 ["2nd"], ["3rd"], ["2nd"], ["3rd"], ["2nd"], ["4th"], ["3rd"], ["5th"],
		 ["4th"], ["2nd"], ["3rd"] ]

# Czerny 599 No. 19
# Measures 1 and 2
start_p8 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations8 = [
	[("C",5)], [("D",5)], [("E",5)], [("F",5)], [("G",5)], [("A",5)], [("B",5)], [("C",6)],
	[("A",5)], [("F",5)], [("C",6)], [("A",5)], [("G",5)], [("G",5)]
]
y0_8 = [ ["1st"], ["2nd"], ["3rd"], ["1st"], ["2nd"], ["3rd"], ["4th"], ["5th"],
         ["3rd"], ["1st"], ["5th"], ["3rd"], ["2nd"], ["1st"] ]

# 去掉了最后那个重复的
# 我猜如果去掉了两个599用例中重复的部分导致能练出来，而留着重复的部分就练不出来的话，blah
start_p8 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations8 = [
	[("C",5)], [("D",5)], [("E",5)], [("F",5)], [("G",5)], [("A",5)], [("B",5)], [("C",6)],
	[("A",5)], [("F",5)], [("C",6)], [("A",5)], [("G",5)]
]
y0_8 = [ ["1st"], ["2nd"], ["3rd"], ["1st"], ["2nd"], ["3rd"], ["4th"], ["5th"],
         ["3rd"], ["1st"], ["5th"], ["3rd"], ["2nd"] ]

# Measures 3 and 4, plus the 1st note in Measure 5
# Force starting with 1
start_p9 = {'1st': 0.01, '2nd': 0.01, '3rd': 0.01, '4th':0.01, '5th': 1.00}
observations9_dedup = [
	[("C",6)], [("B",5)], [("A",5)], [("G",5)], [("F",5)], [("E",5)], [("D",5)], [("C",5)],
	[("E",5)], [("C",5)], [("G",5)], [("E",5)], [("D",5)]
]
y0_9_dedup = [ ["5th"], ["4th"], ["3rd"], ["2nd"], ["1st"], ["3rd"], ["2nd"], ["1st"],
         ["3rd"], ["1st"], ["5th"], ["3rd"], ["2nd"] ]

# Cannot 
start_p10 = {'1st': 1.00, '2nd': 0.01, '3rd': 0.01, '4th':0.01, '5th': 0.01}
observations10 = [
	[("D",5)], [("E",5)], [("D",5)], [("E",5)], [("F",5)], [("D",5)], [("G",5)], [("F",5)],
	[("E",5)], [("F",5)], [("E",5)], [("F",5)], [("G",5)], [("C",6)], [("G",5)], [("E",5)]
]
y0_10 = [
	["1st"], ["2nd"], ["1st"], ["2nd"], ["3rd"], ["1st"], ["4th"], ["3rd"],
	["1st"], ["2nd"], ["1st"], ["2nd"], ["3rd"], ["5th"], ["3rd"], ["2nd"]
]

def GetFingeringDiff(x, y):
	assert len(x) == len(y)
	diff = []
	for idx in range(0, len(x)):
		xx = x[idx]; yy = y[idx];
		if type(xx) == str: # 对xx和yy可能的各种数据类型都考虑到
			if type(yy) == tuple or type(yy) == list:
				if len(yy) == 1 and yy[0] == xx: 
					diff.append(None)
				else: 
					diff.append((xx, yy[0]))
				continue
			elif type(yy) == str:
				if xx == yy:
					diff.append(None)
				else:
					diff.append((xx, yy));
				continue
			else:
				assert False
		elif type(xx) == list or type(xx) == tuple:
			if type(yy) == str:
				if len(xx) == 1 and xx[0] == yy:
					diff.append(None)
				else:
					diff.append((xx[0], yy))
				continue
			else:
				if len(xx) != len(yy):
					diff.append((xx, yy));
				else:
					same = True
					for idx1 in range(0, len(xx)):
						if xx[idx1] == yy[idx1]: continue
						else:
							diff.append((xx, yy));
							same = False;
							break;
					if same:
						diff.append(None)
		else:
			diff.append((xx, yy))
	assert len(diff) == len(x)
	return diff
# 输出的大致形式：[None, (('3rd',), ('2nd',)), None, None, None, None, None, None]

def IsFingeringIdentical(x, y):# 比较器
	if len(x) != len(y):
		return False
	for idx in range(0, len(x)):
		xx = x[idx]; yy = y[idx];
		if type(xx) == str:
			if type(yy) == tuple or type(yy) == list:
				if len(yy) == 1 and yy[0] == xx: continue
				else: 
					return False
			elif type(yy) == str:
				if xx == yy: continue
				else: 
					return False
		elif type(xx) == list or type(xx) == tuple:
			if type(yy) == str:
				if len(xx) == 1 and xx[0] == yy: continue
				else:
					return False
			else:
				if len(xx) != len(yy):
					return False
				else:	
					for idx1 in range(0, len(xx)):
						if xx[idx1] == yy[idx1]: continue
						else: 
							return False
		else:
			return False
	return True

# 隐马尔可夫模型的转移概率矩阵（初始化值，猜的）
transition_probability = {
   '1st' : {'1st': 0.20, '2nd': 0.20, '3rd': 0.20, '4th': 0.20, '5th': 0.20},
   '2nd' : {'1st': 0.20, '2nd': 0.20, '3rd': 0.20, '4th': 0.20, '5th': 0.20},
   '3rd' : {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th': 0.16, '5th': 0.21},
   '4th' : {'1st': 0.23, '2nd': 0.23, '3rd': 0.16, '4th': 0.23, '5th': 0.16},
   '5th' : {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th': 0.16, '5th': 0.21},
}
# for the convenience, chose a iterated one
transition_probability = {
	'4th': {'4th':0.259157,'5th':0.166471,'2nd':0.161543,'3rd':0.175367,'1st':0.237462},
	'5th': {'4th':0.183157,'5th':0.211204,'2nd':0.211204,'3rd':0.183232,'1st':0.211204},
	'2nd': {'4th':0.131409,'5th':0.13899,'2nd':0.228213,'3rd':0.232053,'1st':0.269335},
	'3rd': {'4th':0.15366,'5th':0.197049,'2nd':0.186793,'3rd':0.197049,'1st':0.265449},
	'1st': {'4th':0.106405,'5th':0.230476,'2nd':0.20606,'3rd':0.275574,'1st':0.181486}
}
times = 0 # 看看一共迭代了多少次
def TuningProblem1(training_obs, training_fs, training_starts):

	# Initialize Offender List
	offenders = []
	for i in range(0, len(training_obs)):
		ob = training_obs[i]
		the_list = []
		for j in range(0, len(ob)):
			the_list.append(dict())#append一个空字典，给所有的输入音符留一个位置   
		offenders.append(the_list)

	global g_Y, times
	Xname = ["WY1", "WY2", "WY3", "WY4", "WY5", "BY1", "BY2", "BY3", "BY4", "BY5"]# white。。Black。。
	gradients = [0] * 10
	gradients_t = {
	   '1st' : {'1st': 0, '2nd': 0, '3rd': 0, '4th': 0, '5th': 0},
	   '2nd' : {'1st': 0, '2nd': 0, '3rd': 0, '4th': 0, '5th': 0},
	   '3rd' : {'1st': 0, '2nd': 0, '3rd': 0, '4th': 0, '5th': 0},
	   '4th' : {'1st': 0, '2nd': 0, '3rd': 0, '4th': 0, '5th': 0},
	   '5th' : {'1st': 0, '2nd': 0, '3rd': 0, '4th': 0, '5th': 0},
	}
	global prob_cache
	learning_rate = 1
	learning_rate_t = 0.0001

	for iter in range(0, 99999):
		ok = True
		is_error = []
		wrong_paths = []
		diffs = []
		times += 1

		scratch = []
		for i in range(0, len(training_obs)):
			scratch.append(dict())#不断地建一些空list
		threads = []

		for tidx in range(0, len(training_obs)):
			path, cost1 = viterbiPoly(#用维特比计算最好转换路径和概率值
                                training_obs[tidx],
				states,
				training_starts[tidx],
				transition_probability,
				EmissionProbability_safe
                                )
			if not IsFingeringIdentical(path, training_fs[tidx]):#看是不是和给定的指法一样
				cost2 = GetPathCost(training_obs[tidx], training_fs[tidx],
					training_starts[tidx], transition_probability, EmissionProbability_safe)
				diff = GetFingeringDiff(training_fs[tidx], path);
				for i, x in enumerate(diff):
					if x is None: continue
					else:
						assert type(x) == tuple
						key0 = tuple(x[0])  # Fingering in training data
						val0 = tuple(x[1])  # Fingering generated using VS
						if val0 not in offenders[tidx][i]:
							offenders[tidx][i][val0] = 0
						offenders[tidx][i][val0] += 1
					
				print("Training example %d, not eq training fingering, cost: %g vs %g, diff=%g" %  \
					(tidx, cost1, cost2, cost2-cost1))
				diffs.append(cost2-cost1)
				wrong_paths.append(path)
				is_error.append(True)
				ok = False
			else:
				print("Training example %d,     eq training fingering, cost: %g" %  (tidx, cost1))
				#训练和实际相符，如果这一段被执行，说明所有训练样本全符合，就结束iter，见后面的if ok: break
				is_error.append(False)
				wrong_paths.append(None)
				diffs.append(0)

		if ok:break
 
                
		# Compute Derivatives
		delta = 0.01
                # 初始化transition_probability矩阵
		for x in range(0, len(g_Y)):
			gradients[x] = 0;
		for v in gradients_t.values():
			for vk in v.keys():
				v[vk] = 0

		# Choose one offending data point for stochastic gradient descent
		chosen_idx = -1
		while chosen_idx == -1:
			x = random.randint(0, len(training_obs)-1)
			if is_error[x] == True:
				chosen_idx = x
				break

		# Derivative to Y position of contact points
		for tidx in [chosen_idx]:#只有一个数，就是上面随机选出来的那一个
			sys.stderr.write(str(tidx))
			if is_error[tidx] == False: continue # 为了以防万一的检测
			for x in range(0, len(g_Y)):
				sys.stderr.write(".")
				xold = g_Y[x]
				cost_f_hat  = GetPathCost(training_obs[tidx], wrong_paths[tidx],
					training_starts[tidx], transition_probability, EmissionProbability_safe)
				cost_f_gold = GetPathCost(training_obs[tidx], training_fs[tidx],
					training_starts[tidx], transition_probability, EmissionProbability_safe)

				g_Y[x] = xold + delta
				cost_f_hat_1  = GetPathCost(training_obs[tidx], wrong_paths[tidx],
					training_starts[tidx], transition_probability, EmissionProbability_safe)
				cost_f_gold_1 = GetPathCost(training_obs[tidx], training_fs[tidx],
					training_starts[tidx], transition_probability, EmissionProbability_safe)

				dd_dx = ((cost_f_gold_1 - cost_f_hat_1) - (cost_f_gold - cost_f_hat)) / delta;
				g_Y[x] = xold;#微小调整计算差后再调整回去
				gradients[x] += dd_dx#有不一样就调整gradient
                                # 调整了手接触键的Y坐标g_Y，在EmissionP里取得getPosition中用到
		delta = 0.003
		# Derivative to Transition Probabilities
		for tidx in [chosen_idx]:
			sys.stderr.write(str(tidx))
			if is_error[tidx] == False: continue
			for k, v in gradients_t.items():
				for vk in v.keys():
					sys.stderr.write(".")
					oldv = transition_probability[k][vk]

					cost_f_hat  = GetPathCost(training_obs[tidx], wrong_paths[tidx],
						training_starts[tidx], transition_probability, EmissionProbability_safe)
					cost_f_gold = GetPathCost(training_obs[tidx], training_fs[tidx],
						training_starts[tidx], transition_probability, EmissionProbability_safe)

					transition_probability[k][vk] += delta

					cost_f_hat_1  = GetPathCost(training_obs[tidx], wrong_paths[tidx],
						training_starts[tidx], transition_probability, EmissionProbability_safe)
					cost_f_gold_1 = GetPathCost(training_obs[tidx], training_fs[tidx],
						training_starts[tidx], transition_probability, EmissionProbability_safe)

					dd_dx = ((cost_f_gold_1 - cost_f_hat_1) - (cost_f_gold - cost_f_hat)) / delta;
					transition_probability[k][vk] = oldv
					gradients_t[k][vk] += dd_dx
                #修改这些值.希望用新的矩阵计算出的wrong path的概率值能尽量和正确path的差大一些。希望上面的差值最大化
		for x in range(0, len(g_Y)):
			g_Y[x] += gradients[x] * learning_rate
		for k, v in gradients_t.items():
			for vk in v.keys():
				transition_probability[k][vk] += gradients_t[k][vk] * learning_rate_t
			sum1 = sum(transition_probability[k].values())
			for vk in v.keys():
				transition_probability[k][vk] /= sum1# 保证和还是1

		# Print Offenders
		counts = []
		print("==========Offenders list==============")
		for tidx, o in enumerate(offenders):
			sum0 = 0
			for ety in o:
				sum0 = sum0 + sum(ety.values())
			print("Training example %d has %d offenders" % (tidx, sum0))
			counts.append(sum0)
		print("==========Detailed Offenders list=====")
		for tidx, o in enumerate(offenders):
			if counts[tidx] > 0:
				print("Training example %d (%d):" % (tidx, counts[tidx]))
				for i, ety in enumerate(o):
					if ety is not None:
						correct = training_fs[tidx][i]
						for i1, kv in enumerate(ety.items()):
							print("  Note [%d], %s, %s->%s (%d)" % (i, \
								str(training_obs[tidx][i]), correct, kv[0], kv[1]))
			else:
				print("There are no offenders for training example %d." % tidx)

'''
TuningProblem1(
	[
		observations1, observations2, observations3, 
		observations4, observations5, observations6, 
		observations7,# observations8, observations9_dedup,
		#observations10
	], 
	[y0_1, y0_2, y0_3, y0_4, y0_5, y0_6, y0_7],# y0_8, y0_9_dedup, y0_10],
	[start_p1, start_p2, start_p3, start_p4, start_p5, start_p6, start_p7,]# start_p8, start_p9, start_p10]
)
'''

#训练完的结果
'''
g_Y = (
    [27.169387530125277, 29.06759800649958, 45.40988125058604, 29.754358237078122,
    30.75211529102372, 70.0, 71.24985203290066, 78.92078510053643, 75.00000651514469,
    69.99999999999787]
       )

#第一次算出来的一个结果
transition_probability = (
    {
     '2nd': {'2nd': 0.2308866830680898, '1st': 0.26081696750252925, '3rd': 0.2317602722400613, '4th': 0.13591770715941798, '5th': 0.1406183700299016},
     '1st': {'2nd': 0.21152041692098747, '1st': 0.18164934010081007, '3rd': 0.2609123246568355, '4th': 0.12138380948091372, '5th': 0.22453410884045324},
     '3rd': {'2nd': 0.1655861707403901, '1st': 0.2639830338432041, '3rd': 0.19733736480277086, '4th': 0.1757560658108653, '5th': 0.1973373648027696},
     '4th': {'2nd': 0.15793136743892283, '1st': 0.2294911881403911, '3rd': 0.20440371040249108, '4th': 0.25569059359518614, '5th': 0.15248314042300884},
     '5th': {'2nd': 0.21164703753610187, '1st': 0.21164703753610187, '3rd': 0.18151751679860043, '4th': 0.18354137059309386, '5th': 0.21164703753610187}
     }
    )

'''
'''
g_Y = ([27.204411035247134, 28.848954304451553, 45.484231960504786, 29.774043653020783,
        30.532821319773973, 70.00000000000178, 70.86186944675781, 79.14954852997155,
        75.00000651459757, 69.99999999999929])
transition_probability = (
    {
        '5th': {'5th': 0.21119403245171425, '2nd': 0.21119403245171425,
                '1st': 0.21119403245171425, '3rd': 0.18160699924880142,
                '4th': 0.18481090339605583},
        '2nd': {'5th': 0.14051160621019307, '2nd': 0.2307113834667742,
                '1st': 0.26138688169950985, '3rd': 0.23158560814417523,
                '4th': 0.13580452047934766},
        '1st': {'5th': 0.22566881143549383, '2nd': 0.21223333343684037,
                '1st': 0.1818398498542585, '3rd': 0.2604254602045469,
                '4th': 0.11983254506886032},
        '3rd': {'5th': 0.19759563283068862, '2nd': 0.16589539120965788,
                '1st': 0.26361002659544713, '3rd': 0.19759563283068934,
                '4th': 0.17530331653351702},
        '4th': {'5th': 0.15374268363255747, '2nd': 0.1579801657919586,
                '1st': 0.22959962000466166, '3rd': 0.20358257886356843,
                '4th': 0.2550949517072539}
        }
    )
    '''

my_start_p = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
my_start_p_sitei = {'1st': 0.9, '2nd': 0.025, '3rd': 0.025, '4th':0.025, '5th': 0.025}
my_observation1 = [# somewhere in time
	[("C",4)], [("D",4)],[("E",4)],[("B",4)],[("C",5)],[("B",4)],[("G",4)],[("D",4)],
	[("E",4)],[("A",3)],[("B",3)],[("C",4)],[("D",4)], [("E",4)],[("D",4)],[("E",4)],
        [("D",4)],[("A",4)],[("B",3)],[("B",4)],[("G",4)] ]
my_observation2 = [# No.11 Mozart:Turkei
	[("B",4)], [("A",4)],[("#G",4)],[("A",4)],[("C",5)],[("D",5)],[("C",5)],[("B",4)],
	[("C",5)],[("E",5)],[("F",5)],[("E",5)],[("#D",5)], [("E",5)],[("B",5)],[("A",5)],
        [("#G",5)],[("A",5)],[("B",5)],[("A",5)],[("#G",5)],[("A",5)],[("C",6)] ]

path1, cost1 = viterbiPoly(#应该是用维特比计算最好转换路径和概率值
                                my_observation2,
				states,
				my_start_p,
				transition_probability,
				EmissionProbability_safe
                                )
path2, cost2 = viterbiPoly(#看一下是不是真的不能全部解决offender::解决了的！
                                observations6,
				states,
				my_start_p,
				transition_probability,
				EmissionProbability_safe
                                )
dd = GetFingeringDiff(path2, y0_6)
print(path1)
print('-------')
print(dd)
exit(0)

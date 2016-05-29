
g_Y = (
    [27.169387530125277, 29.06759800649958, 45.40988125058604, 29.754358237078122,
    30.752115229102372, 70.0, 71.24985203290066, 78.92078510053643, 75.00000651514469,
    69.99999999999787]
       )

transition_probability = (
    {
     '2nd': {'2nd': 0.2308866830680898, '1st': 0.26081696750252925, '3rd': 0.2317602722400613, '4th': 0.13591770715941798, '5th': 0.1406183700299016},
     '1st': {'2nd': 0.21152041692098747, '1st': 0.18164934010081007, '3rd': 0.2609123246568355, '4th': 0.12138380948091372, '5th': 0.22453410884045324},
     '3rd': {'2nd': 0.1655861707403901, '1st': 0.2639830338432041, '3rd': 0.19733736480277086, '4th': 0.1757560658108653, '5th': 0.1973373648027696},
     '4th': {'2nd': 0.15793136743892283, '1st': 0.2294911881403911, '3rd': 0.20440371040249108, '4th': 0.25569059359518614, '5th': 0.15248314042300884},
     '5th': {'2nd': 0.21164703753610187, '1st': 0.21164703753610187, '3rd': 0.18151751679860043, '4th': 0.18354137059309386, '5th': 0.21164703753610187}
     }
    )

path, cost1 = viterbiPoly(#应该是用维特比计算最好转换路径和概率值
                                observations1,
				states,
				start_p1,
				transition_probability,
				EmissionProbability_safe
                                )
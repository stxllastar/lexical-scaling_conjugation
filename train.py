import numpy as np
from sys import exit
from scipy.optimize import minimize
from matplotlib.pyplot import plot, show
from datetime import datetime
from sys import argv

#This is a revised version of the scaled-weights MaxEnt model described in Hughto et al. (2019), allowing the stem and suffix to receive different priors.

#####USER SETTINGS#####   
#General parameters:
METHOD = argv[1]#Choose from the set {GD, GD_CLIP, and L-BFGS-B}
NEG_WEIGHTS = bool(int(argv[2])) #Set this to True if you're cool with negative weights
NEG_SCALES = bool(int(argv[3])) #Set this to True if you're cool with negative scales
LAMBDA1 = float(argv[4]) #Weight for regularization of general weights
LAMBDA2 = float(argv[5]) #Weight for regularization of stems
LAMBDA3 = float(argv[6]) #Weight for regularization of suffixes

L2_PRIOR = bool(int(argv[7])) #Right now, this only works with L-BFGS-B
INIT_WEIGHT = float(argv[8]) #Initial weights (if RAND_WEIGHTS = False)
RAND_WEIGHTS = bool(int(argv[9])) #Makes initial weights randoms ints between 0-10

#Params for gradient descent:
ETA = float(argv[10])     #Learning rate
EPOCHS = int(argv[11])  #Iterations through whole dataset

#Other user settings:
TD_FILE = argv[12] #Training data (needs to include directory info if it's not in the same one)

language = argv[13]
OUTPUT_DIR = argv[14] #output directory (needs to include slashes at the end!)

#####FUNCTIONS##### 
def get_predicted_probs (weights, scales, viols):
    #First we need to add up the scales for the morphemes in each datum:
    scales_by_datum = np.array([np.sum(scales[datum2morphs[datum]], axis=0)\
                                        for datum in range(len(p))])
    #Then we add the scales to the weights:
    scaledWeights_by_datum = scales_by_datum + weights 
    
    #Simple MaxEnt stuff:
    harmonies = np.sum(viols * scaledWeights_by_datum, axis=1)
    eharmonies = np.exp(-1 * harmonies)
    Z_by_UR = np.array([sum(eharmonies[ur2data[URs[datum]]]) \
                            for datum, viol in enumerate(viols)])
    probs = eharmonies/Z_by_UR

    return probs
    
def get_nonce_probs (weights, viols):
    harmonies = viols.dot(weights)
    eharmonies = np.exp(-1 * harmonies)
    Z_by_UR = np.array([sum(eharmonies[ur2data[URs[datum]]]) \
                        for datum, viol in enumerate(viols)])
    probs = eharmonies/Z_by_UR
    
    return probs    

def grad_descent_update (weights, viols, td_probs, scales, eta=.05):
    #Dimensions (repeated for new variables throughout):
    #Weights -> C (where C is the # of constraints)
    #td_probs -> D (where D is the # of data)
    #Viols -> DxC
    #Scales -> MxC (where M is the # of morphemes)
    
    #Forward pass (learner's expected probabilities):
    le_probs = get_predicted_probs (weights, scales, viols) #(D)
    
    #Backward pass:
    TD_byDatum = viols.T * td_probs #Violations by datum present in the training data (CxD)
    LE_byDatum = viols.T * le_probs #Violations by datum expected by the learner (CxD)
    
    #Convert the expected violations by datum to expected violations by morpheme 
    #(this could probably be more efficient):
    TD_byMorph = np.zeros(scales.shape) #Violations by morph present in the training data (MxC)
    LE_byMorph = np.zeros(scales.shape) #Violations by morph expected by the learner (MxC)
    for datum_index, datum2morph in enumerate(datum2morphs):
        for morph in datum2morph:
            TD_byMorph[morph] += TD_byDatum.T[datum_index]
            LE_byMorph[morph] += LE_byDatum.T[datum_index] 
    
    #The part of the gradients from the log loss is TD-LE (obs. - exp.)
    c_gradients = np.sum(TD_byDatum, axis=1) - np.sum(LE_byDatum, axis=1) #(C)
    s_gradients = TD_byMorph - LE_byMorph #(MxC)
    
    #Update based on log loss:
    almost_new_weights = weights - (c_gradients * eta)#aka w^{k+1/2} (C)
    almost_new_scales = scales - (s_gradients * eta)#(MxC)
    
    #Updates based on prior:
    if "CLIP" in METHOD:
        #With clipping (Tsuruoka et al. 2009):
        new_weights = []
        for w_kPlusHalf in almost_new_weights:
            if w_kPlusHalf > 0:
                #If the weight is above zero, 
                #don't let the prior take it below zero.
                w_kPlus1 = max([0, w_kPlusHalf - ((LAMBDA1)*eta)])
            elif w_kPlusHalf < 0:
                #If the weight is below zero, don't let the prior take it
                #above zero.
                w_kPlus1 = min([0, w_kPlusHalf + ((LAMBDA1)*eta)])
            else:
                #If the weight is exactly zero, don't change it.
                w_kPlus1 = w_kPlusHalf
            new_weights.append(w_kPlus1)
        
    
        new_scales = []
        for i, morpheme_scales in enumerate(almost_new_scales):
    
            lam = LAMBDA2 if i in stem_indices else LAMBDA3
        
            
            this_morphs_scales = []
            for s_kPlusHalf in morpheme_scales:
                if s_kPlusHalf > 0:
                    #If the scale is above zero, 
                    #don't let the prior take it below zero.
                    s_kPlus1 = max(0, s_kPlusHalf - lam * eta)
                elif s_kPlusHalf < 0:
                    #If the scale is below zero, don't let the prior take it
                    #above zero.
                    s_kPlus1 = min(0, s_kPlusHalf + lam * eta)

                else:
                    #If the scale is exactly zero, don't change it.
                    s_kPlus1 = s_kPlusHalf
                this_morphs_scales.append(s_kPlus1)
            new_scales.append(this_morphs_scales)
            
        new_weights = np.array(new_weights)
        new_scales = np.array(new_scales)
    else:
        #Without clipping:
        prior1_gradient = (LAMBDA1 * weights) #(C)
        prior2_gradient = (LAMBDA2 * stem_scales) #(MxC)
        prior3_gradient = (LAMDA3 * suffix_scales)
        
        new_weights =  almost_new_weights + prior1_gradient * eta #aka w^{k+1}
       
        #separate stems and scales
        new_scales = almost_new_scales.copy()
        new_scales[stem_indices]   += eta * prior2_gradient
        new_scales[suffix_indices] += eta * prior3_gradient


    #And only have negative stuff if we want to allow it:
    if not NEG_WEIGHTS:
        new_weights = np.maximum(new_weights, 0)
    if not NEG_SCALES:
        new_scales = np.maximum(new_scales, 0)
        
    return new_weights, new_scales

def objective_function (weightsAndScales, viols, td_probs, areScalesFlat=True):
    #This function is used by the LBFGSB optimizer:

    to_return = ()

    #Handle "weightsAndScales" differently, depending on the areScalesFlat parameter:
    if areScalesFlat:
        weights, scales = weightsAndScales[:len(w)], weightsAndScales[len(w):]
        scales = np.reshape(scales, s.shape)
    else:
        weights = weightsAndScales[0]
        scales = weightsAndScales[1:]
        
    #Regular log loss stuff:
    le_probs = get_predicted_probs(weights, scales, viols)
    log_probs = np.log(le_probs)
    loss = (-1.0 * np.sum(td_probs*log_probs)) #log loss
    to_return += (loss,) #keep track of loss separate from pior

    if L2_PRIOR:
        loss += (np.sum(weights**2) * LAMBDA1) + \
                (np.sum(scales[stem_indices]**2) * LAMBDA2) + \
                (np.sum(scales[suffix_indices]**2) * LAMBDA3)

    else:
        prior_weights = np.sum(np.abs(weights)) * LAMBDA1
        prior_stem    = np.sum(np.abs(scales[stem_indices])) * LAMBDA2
        prior_suffix  = np.sum(np.abs(scales[suffix_indices])) * LAMBDA3
        prior_total   = prior_weights + prior_stem + prior_suffix
        
        to_return += (prior_total,)	#keep track of prior separately
        loss += prior_total
        to_return += (loss,) 	#full objective fnc.
    if "GD" in METHOD:
        return to_return
    else:
        return to_return[-1]
    
#####PROCESS LEARNING DATA##### 
tableaux_file = open(TD_FILE, "r") #training data file

morphemes = tableaux_file.readline().rstrip().split("\t") #list of the morphemes

#separate this into stem and suffix
stem_indices = [i for i, m in enumerate(morphemes) if m.startswith("s")]
suffix_indices = [i for i, m in enumerate(morphemes) if m.startswith("p")]


c_names = [c for c in tableaux_file.readline().rstrip().split("\t") if c != ''] #list of constraints
v = [] #This will store each candidate's violation profile
p = [] #This will store each candidate's conditional probability, i.e. Pr(SR|UR)
datum2morphs = []
last_in = ""
URs = []
SRs = []
for row in tableaux_file.readlines():
    columns = row.rstrip().split("\t")
    my_in, my_out, my_prob = columns[:3]
    if my_in == "":
        my_in = last_in
    my_viols = [int(viol) for viol in columns[3:]]
    my_prob = float(my_prob)

    v.append(my_viols)
    p.append(my_prob) 
    URs.append(my_in)
    SRs.append(my_out)
    datum2morphs.append([morphemes.index(morph) for morph in my_in.split("$")[3:]])
    
    last_in = my_in

#Create a dictionary for efficiently finding Z's:
ur2data = {form:[] for form in URs}
for datum_index, ur in enumerate(URs):
    ur2data[ur].append(datum_index)

#All the arrays we need: 
start_time = ",".join(str(datetime.now()).split(":"))
v = np.array(v)                          #Constraint violations
if RAND_WEIGHTS:
    w = np.random.uniform(low=0.0, high=10.0, size=len(v[0]))
    weights_file = open(OUTPUT_DIR+start_time+" init_weights.txt", "w")
    for c, name, in enumerate(c_names):
        weights_file.write(name+"\t"+str(w[c])+"\n")
    weights_file.close()
else:
    w = np.array([INIT_WEIGHT for c in v[0]])      #Constraint weights                                           
p = np.array(p)                                 #Learning data probs
s = np.array([[0.0 for c in c_names] for morph in morphemes]) #Initial scales 

stem_scales   = s[stem_indices]      # rows for stems
suffix_scales = s[suffix_indices]    # rows for suffixes

all_params = np.concatenate((w,np.ndarray.flatten(s)))

#####LEARNING##### 
if "GD" in METHOD:
    init_loss = objective_function(all_params, v, p)[0]
    loss_tracker = []
    min_loss = 1000000
    best_weights = []
    best_scales = []
    best_epoch = -1
    for ep in range(EPOCHS):
        full_params = np.concatenate((np.array([w]), s))
        this_loss = objective_function(full_params, v, p, False)
        loss_tracker.append(this_loss)
        if this_loss[-1] < min_loss:
            best_weights = w
            best_scales = s
            best_epoch = ep
            min_loss = this_loss[-1]
        if ep % 100 == 0:
            print("Epoch: "+str(ep)+"\tLoss, Prior, Obj: "+str(this_loss))
            print("\tweights: "+str(w))
            
        w, s = grad_descent_update (w, v, p, s, eta=ETA)
    final_weights = w
    final_params = np.concatenate((np.array([final_weights]), s))
    final_scales = s
    
    print("Beginning loss: ", init_loss)
    print("Final loss: ", objective_function(final_params, v, p, False))
else:
    init_loss = objective_function(all_params, v, p)
    if NEG_WEIGHTS:
        min_w = None
    else:
        min_w = 0.0
    final_params = minimize(objective_function, all_params, args=(v, p), method=METHOD, bounds=[(min_w, None) for x in all_params])['x']
    final_weights = final_params[:len(w)]
    
    final_scales = np.zeros_like(s)
    # Flattened stem and suffix arrays

    C = len(c_names)
    stem_size = len(stem_indices) * C
    suffix_size = len(suffix_indices) * C

    stem_flat   = final_params[len(w) : len(w) + stem_size].reshape(len(stem_indices), C)
    suffix_flat = final_params[len(w) + stem_size : ].reshape(len(suffix_indices), C)

    final_scales[stem_indices]   = stem_flat
    final_scales[suffix_indices] = suffix_flat
    
    #this was the original code
    #final_scales = np.reshape(final_params[len(w):], s.shape)
    
    print("Beginning loss: ", init_loss)
    print("Final loss: ", objective_function(final_params, v, p))

#####OUTPUT##### 

#Print final state of the grammar:
print("Printing grammar to file...")
WEIGHT_file = open(OUTPUT_DIR+"outputWeights_"+language+f"_epochs={EPOCHS}"+f"_lr={ETA}"+"_C="+f"{LAMBDA1}_{LAMBDA2}_{LAMBDA3}"+".csv", "w")
WEIGHT_file.write("Morpheme,"+",".join(c_names)+"\n")
WEIGHT_file.write("General,"+",".join([str(fw) for fw in final_weights])+"\n")

for m_index, this_morph in enumerate(morphemes):
    m_weights = final_scales[m_index]
    WEIGHT_file.write(this_morph+","+",".join([str(mw) for mw in m_weights])+"\n")
WEIGHT_file.close()

#Print nonce judgments:
print("Printing nonce judgments to file...")
NONCE_file = open(OUTPUT_DIR+"nonceProbs_"+language+f"_epochs={EPOCHS}"+f"_lr={ETA}"+"_C="+f"{LAMBDA1}_{LAMBDA2}_{LAMBDA3}"+".csv", "w")
NONCE_file.write("Input,Output,ObservedProb,ExpectedProb\n")
my_nonce_probs = get_nonce_probs(final_weights, v)
for datum_index, datum_prob in enumerate(my_nonce_probs):
    ur_string = URs[datum_index].split("$")[0]
    NONCE_file.write(ur_string+","+SRs[datum_index]+","+\
                        str(p[datum_index])+","+str(datum_prob)+"\n")   
NONCE_file.close()

#Print training data judgments:
print("Printing training data judgments to file...")
TDJ_file = open(OUTPUT_DIR+"tdProbs_"+language+f"_epochs={EPOCHS}"+f"_lr={ETA}"+"_C="+f"{LAMBDA1}_{LAMBDA2}_{LAMBDA3}"+".csv", "w")
TDJ_file.write("Input,Output,ObservedProb,ExpectedProb\n")
my_td_probs = get_predicted_probs(final_weights, final_scales, v)
for datum_index, datum_prob in enumerate(my_td_probs):
    ur_string = URs[datum_index].split("$")[0]
    TDJ_file.write(ur_string+","+SRs[datum_index]+","+\
                        str(p[datum_index])+","+str(datum_prob)+"\n")   
TDJ_file.close()

if "GD" in METHOD:
    #Print info for the epoch with the lowest loss:
    print("Printing best epoch data to file...")
    bestEP_file = open(OUTPUT_DIR+"bestEpoch_"+language+f"_epochs={EPOCHS}"+f"_lr={ETA}"+"_C="+f"{LAMBDA1}_{LAMBDA2}_{LAMBDA3}"+".txt", "w")
    bestEP_file.write("Best epoch was "+str(best_epoch)+"\n")
    bestEP_file.write("Loss at best epoch was "+str(min_loss)+"\n")
    bestEP_file.write("Weights:\n")
    for c_i, bw in enumerate(best_weights):
        bestEP_file.write(c_names[c_i]+"\t"+str(bw)+"\n")
    bestEP_file.write("Scales:\n")
    for m_i, bs in enumerate(best_scales):
        bestEP_file.write(morphemes[m_i]+"\n")
        for c_i, this_s in enumerate(bs):
            bestEP_file.write("\t"+c_names[c_i]+"\t"+str(this_s)+"\n") 
    bestEP_file.close()

#All done!
print("All done!") 

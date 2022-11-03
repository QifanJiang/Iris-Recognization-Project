import matplotlib.pyplot as plt

# Calculate the CRR
def CRRCalculation(df_min, method):
    count = 0
    correct_count = 0
    results = []
    for i in df_min[method]:
        count += 1
        if (count-1)//4 == (i-1)//21:
            correct_count += 1
            results.append(1)
            continue
        results.append(0)
    return (correct_count/count)*100, results

def PerformanceEvaluation(df_min_list,df_minmeasure_list,n_components_list):
    # CRR using the dimensionality of 107
    CRR_L1,_ = CRRCalculation(df_min_list[-1],"L1")
    CRR_L2,_ = CRRCalculation(df_min_list[-1],"L2")
    CRR_Cosine,results = CRRCalculation(df_min_list[-1],"Cosine")
    print("CRR by L1 distance measure: {:.3f}".format(CRR_L1))
    print("CRR by L2 distance measure: {:.3f}".format(CRR_L2))
    print("CRR by Cosine similarity measure: {:.3f}".format(CRR_Cosine))

    # Plot and store the graph of CRR vs Dimensionality on Cosine similarity
    CRR_Cosine_list = []
    for df_min in df_min_list:
        CRR,_ = CRRCalculation(df_min,"Cosine")
        CRR_Cosine_list.append(CRR)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(n_components_list,CRR_Cosine_list)
    plt.xlabel("Dimensionality of the feature vector")
    plt.ylabel("Correct Recognition Rate (%)")
    plt.title("CRR vs Dimensionality on Cosine similarity")

    FNMRs = []
    FMRs = []    
    thresholds = [0.446, 0.472, 0.502]
    dfs = df_minmeasure_list[-1]['Cosine']
    num_true = results.count(1)
    num_false = results.count(0)

    # Calculate FNMR and FMR
    for threshold in thresholds:
        tf=0
        ft=0
        for i in range(len(dfs)):
            # if the result should be True but we mark it False
            if dfs[i+1] < threshold:
                if results[i] == 0:
                    tf += 1
            # if the result should be False but we mark it True
            if dfs[i+1] > threshold:
                if results[i] == 1:
                    ft += 1
        FNMRs.append(ft/num_true)
        FMRs.append(tf/num_false)
    print("Threshold {}: FNMR = {}, FMR = {}".format(thresholds[0],FNMRs[0],FMRs[0]))
    print("Threshold {}: FNMR = {}, FMR = {}".format(thresholds[1],FNMRs[1],FMRs[1]))
    print("Threshold {}: FNMR = {}, FMR = {}".format(thresholds[2],FNMRs[2],FMRs[2]))
    plt.subplot(1,2,2)
    plt.plot(FNMRs, FMRs)
    plt.xlabel("False Match Rate (%)")
    plt.ylabel("False Non-Match Rate (%)")
    plt.title("ROC")
    plt.savefig("plot.png")
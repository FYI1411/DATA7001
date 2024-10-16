#!/usr/bin/env python3
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib import style
import math 
from matplotlib.lines import Line2D



data_directory = "vis_project_data"

# Default weights; used if not set
default_weights = np.array([-1, 0, 1, 2, 2, 3, 5])
soc_weights = np.array([-1, 0, 1, 2, 2, 3, 5])
cog_weights = np.array([0, -1, 1, 2, 2, 3, 5])
session_types = ["cog", "so"]
whatdoors = ["indoor", "outdoor"]
whichs = ["base", "inter"]
combined_scenarios = [
    (ses_type, whatdoor, which)
    for ses_type in session_types
    for whatdoor in whatdoors
    for which in whichs
]

"""
residual plot

Overall score / time ?
effect on outdoor/indoor on same plot maybe?^
effect on cog/soc as well

box plots?
"""

################################################################################
def combined_score(filename, weights):
    """Calculates the 'score' for a single session/file.
    Assumes total session duration is 360s, otherwise returns 'nan'.
    This could be modified simply to also return other details of the session."""
    with open(filename, "r") as file:
        score = 0.0
        total_duration = 0.0
        t_end_prev = 0.0
        for count, line in enumerate(file.readlines()):
            # print(count, line)
            data = line.split(",", 4)
            if count == 0:
                continue
            if line[0] == "*":
                break

            t_catagory = int(data[0])
            t_beg = int(data[1])
            t_end = int(data[2])

            if t_beg != t_end_prev:
                print("Error, missing time stamp?")
            t_end_prev = t_end

            assert t_end >= t_beg
            if count == 1:
                assert t_beg == 0

            duration = float(t_end - t_beg)
            total_duration += duration
            score += weights[t_catagory - 1] * duration
        return score / total_duration
        return score if np.abs(total_duration - 1.0) < 1.0e-5 else np.nan
    


################################################################################
def unique_pairs():
    """Returns list of unique ca/peer pairs"""
    all_files = glob.glob(data_directory + "/*.dtx")
    list = []
    for file in all_files:
        t = file.split("-")
        list.append([t[4], t[5]])
    return np.unique(list, axis=0)

################################################################################
        
def scores_list(ca, peer, ses_type, whatdoor, cognitive_weights=cog_weights, social_weights=soc_weights):
    """ When given a ca, peer, session type (cog/so) and a whatdoor (indoor/outdoor), will
        return a NP.ARRAY of the scores corresponding to the specifications """
    b_scores, i_scores = [], []
    for which in ["base", "inter"]: 
        files = glob.glob(data_directory + "/" + f"{ses_type}-*-{which}-*-{ca}-{peer}-{whatdoor}.dtx")
        for file in files:
            # date = file.title().split("-")[1]
            if ses_type == "so":
                weights = social_weights
            else:
                weights = cognitive_weights
            tmp_score = combined_score(file, weights)
            if not np.isnan(tmp_score):
                if which == 'base':
                    b_scores.append(tmp_score)

                else:
                    i_scores.append(tmp_score)

    b_scores, i_scores = np.array(b_scores), np.array(i_scores)         
    return b_scores, i_scores

def trained_untrained_scores(ca, trained, ses_type, whatdoor, cognitive_weights=cog_weights, social_weights=soc_weights):
    b_scores, i_scores = [], []
    if trained:
        for which in ["base", "inter"]: 
            for peer in ["ulrich", "viola", "wendy", "xavier"]:
                files = glob.glob(data_directory + "/" + f"{ses_type}-*-{which}-*-{ca}-{peer}-{whatdoor}.dtx")
                for file in files:
                    # date = file.title().split("-")[1]
                    if ses_type == "so":
                        weights = social_weights
                    else:
                        weights = cognitive_weights
                    tmp_score = combined_score(file, weights)
                    if not np.isnan(tmp_score):
                        if which == 'base':
                            b_scores.append(tmp_score)

                        else:
                            i_scores.append(tmp_score)
    else:
        for which in ["base", "inter"]:
            for peer in ['lydia', 'mario', 'nellie', 'oscar']:
                files = glob.glob(data_directory + "/" + f"{ses_type}-*-{which}-*-{ca}-{peer}-{whatdoor}.dtx")
                for file in files:
                    # date = file.title().split("-")[1]
                    if ses_type == "so":
                        weights = social_weights
                    else:
                        weights = cognitive_weights
                    tmp_score = combined_score(file, weights)
                    if not np.isnan(tmp_score):
                        if which == 'base':
                            b_scores.append(tmp_score)

                        else:
                            i_scores.append(tmp_score)

    b_scores, i_scores = np.array(b_scores), np.array(i_scores)         
    return b_scores, i_scores


def trained_untrained_bi(ca_peer_list, ses_type, whatdoor, cognitive_weights=cog_weights, social_weights=soc_weights):
    """ When given all the ca/peer combinations and one session type + whatdoor, will
        return 8 variables corresponding to:
        0: list of trained base scores (for CA1, CA2, CA3, CA5)
        1: list of the trained base STANDARD ERRORS (sem = std/sqrt(n)) for the same CAs
        2: list of trained inter scores
        3: list of trained inter standard errors
        4: list of untrained base scores
        5: list of untrained base standard errors
        6: list of untrained inter scores
        7: list of untrained inter standard errors """
    c = {"albert": 0, "barry": 0, "chris": 0, "dana": 0} 
    tra_b, tra_b_e, tra_i, tra_i_e = c.copy(), c.copy(), c.copy(), c.copy()
    un_b, un_b_e, un_i, un_i_e = c.copy(), c.copy(), c.copy(), c.copy()
    
    for ca, peer in ca_peer_list:
        if ca != "ellie":  # ignoring Ellie (CA4)
            b_scores, i_scores = scores_list(ca, peer, ses_type, whatdoor, cognitive_weights, social_weights)
            b_mean, i_mean = b_scores.mean(), i_scores.mean()
            b_sem, i_sem = stats.sem(b_scores, ddof=1), stats.sem(i_scores, ddof=1)  # "corrected" sdev to get sem

            if peer in ['lydia', 'mario', 'nellie', 'oscar', 'peter']:
                un_b[ca] = b_mean
                un_i[ca] = i_mean
                un_b_e[ca], un_i_e[ca] = b_sem, i_sem
                
            else:
                tra_b[ca] = b_mean
                tra_i[ca] = i_mean
                tra_b_e[ca], tra_i_e[ca] = b_sem, i_sem
    
    # print("trained: ", trained_b)
    # print("\nuntrained: ", untrained_b)
    return(tra_b, tra_b_e, tra_i, tra_i_e, un_b, un_b_e, un_i, un_i_e)


def split_results(ca_peer_list, ses_type, whatdoor, cog_weights, soc_weights):
    """ helper method for the next function, can ignore """
    # output = tra_b, tra_b_e, tra_i, tra_i_e, un_b, un_b_e, un_i, un_i_e
    tra_b, tra_b_e, tra_i, tra_i_e, un_b, un_b_e, un_i, un_i_e \
    = trained_untrained_bi(ca_peer_list, ses_type, whatdoor, cog_weights, soc_weights)
    tra_b_mean = np.array(list(tra_b.values())).mean()
    tra_b_sem = stats.sem(np.array(list(tra_b.values())), ddof=1)
    tra_i_mean = np.array(list(tra_i.values())).mean()
    tra_i_sem = stats.sem(np.array(list(tra_i.values())), ddof=1)
    un_b_mean = np.array(list(un_b.values())).mean()
    un_b_sem = stats.sem(np.array(list(un_b.values())), ddof=1)
    un_i_mean = np.array(list(un_b_e.values())).mean()
    un_i_sem = stats.sem(np.array(list(un_b_e.values())), ddof=1)
    return [tra_b_mean, tra_b_sem, tra_i_mean, tra_i_sem, un_b_mean, un_b_sem, un_i_mean, un_i_sem]

def overall_results(ca_peer_list, cognitive_weights=cog_weights, social_weights=soc_weights):
    """ When given the list of ca/peers, will return 8 lists. Each list contains FOUR
        objects, corresponding to [indoor-cog, outdoor-cog, indoor-so, outdoor-so]. For example
        0 is a list of the average trained base scores -> the list contains [average (over all 
        CAs) trained_base_scores for indoor-cog, average_trained_base_scores for outdoor-cog,
        average_trained_base_scores for indoor-so, average_trained_base_scores for outdoor-so]. 
        The rest of the lists contain:

        0: list of 4 AVERAGE trained base scores (all ca/peers), corresponding to the order above
        1: list of 4 average trained base STANDARD ERRORS (sem = std/sqrt(n)) for the same CAs
        2: list of 4 average trained inter scores
        3: list of 4 average trained inter standard errors
        4: list of 4 average untrained base scores
        5: list of 4 average untrained base standard errors
        6: list of 4 average untrained inter scores
        7: list of 4 average untrained inter standard errors 
    """
    tra_b, tra_b_e, tra_i, tra_i_e, un_b, un_b_e, un_i, un_i_e = [],[],[],[],[],[],[],[]
    for ses in ["cog", "so"]:
        for door in ["indoor", "outdoor"]:
            output = split_results(ca_peer_list, ses, door, cognitive_weights, social_weights)
            tra_b.append(output[0]), tra_b_e.append(output[1]), tra_i.append(output[2]), tra_i_e.append(output[3])
            un_b.append(output[4]), un_b_e.append(output[5]), un_i.append(output[6]), un_i_e.append(output[7])
    return tra_b, tra_b_e, tra_i, tra_i_e, un_b, un_b_e, un_i, un_i_e
    # in order of 


def interleave(a,b):
    """ Random helper method i used to change the order 
    FROM: cog, in base -> cog, out base-> so, in base -> so, out base -> cog, in inter -> cog, out inter -> so, in inter -> so, out inter
    TO: cog, in base -> cog, in inter -> cog, out base -> cog, out inter -> etc.
    """
    x = []
    for i in range(len(a)):
        x.append(a[i])
        x.append(b[i])
    return x
    
def clamp(n, minn, maxn):
    """ Random helper method i used to hold a value within a min and a max (i used this to
    lock my alpha values for colour between 0 and 1 bc they have to be)"""
    return max(min(maxn, n), minn)


#################################################################################



def grouped_bar_one(before, after, bef_e, aft_e):
    """ This one is a grouped bar graph that gradients colour with statistical significance
        NOTE: this is NOT good (i used this one as an example of something i tried that DIDNT
        WORK)"""
    bef, bef_e = list(before.values()), list(bef_e.values())
    aft, aft_e = list(after.values()), list(aft_e.values())
    b, a = np.array(bef), np.array(aft)
    bef_e.append(stats.sem(b, ddof=1))
    bef.append(b.mean())
    aft_e.append(stats.sem(a, ddof=1))
    aft.append(a.mean())

    colours = []
    for i in range(5):
        # using c as a measure of statistical signifiance, gradient from really green -> really red
        c = (aft[i] - aft_e[i]) - (bef[i] + bef_e[i])
        c = c/((bef[i]+aft[i])/2)

        if c > 1.5:
            colours.append("#00FF00")
        elif c > 1:
            colours.append("#66ff00")
        elif c > 0.5:
            colours.append("#99ff00")
        elif c > 0: 
            colours.append("#ccff00")
        elif c < 0:
            colours.append("#FFCC00")
        elif c < -0.5:
            colours.append("#ff9900")
        elif c < -1:
            colours.append("#FF3300")
        elif c < -1.5:
            colours.append("#FF0000")
    
    x = np.arange(5)
    plt.bar(x-0.2, bef, yerr=bef_e, capsize=5, width=0.4)
    plt.bar(x+0.2, aft, yerr=aft_e, capsize=5, width=0.4, color=colours, alpha = 1)
    plt.title('Effect of Training on Outdoor Social Interactions\n')
    plt.xticks(x, ["base       inter\n CA1", "base       inter\n CA2", "base       inter\n CA3", "base       inter\n CA5", "base       inter\n All CAs"]) 
    plt.xlabel("Dyad") 
    plt.ylabel("Social Interaction Score")
    plt.legend(["base", "inter"])
    plt.show()

def grouped_bar_two(before, after, bef_e, aft_e):
    """ This one is a much better grouped bar graph, that uses SHADE to indicate statistical
        significance and COLOUR to indicate direction of change (e.g., dark green would indicate
        statistically significant INCREASE, lighter green = less significant increase, dark red
        is significant decrease)"""
    bef, bef_e = list(before.values()), list(bef_e.values())
    aft, aft_e = list(after.values()), list(aft_e.values())
    b, a = np.array(bef), np.array(aft)
    bef_e.append(stats.sem(b, ddof=1))
    bef.append(b.mean())
    aft_e.append(stats.sem(a, ddof=1))
    aft.append(a.mean())
    
    
    colours = []
    for i in range(5):
        d = aft[i] - bef[i]
        if d >= 0:
            c = (aft[i] - aft_e[i]) - (bef[i] + bef_e[i])
            c = c/((bef[i]+aft[i])/4)
            # [r,g,b,alpha] used alpha to change shade
            colours.append([0,1,0,clamp(c, 0.3, 1)])
        else:
            c = (bef[i] - bef_e[i]) - (aft[i] + aft_e[i])
            c = c/((bef[i]+aft[i])/5)
            print(c)
            colours.append([1,0,0,clamp(c, 0.3, 1)])
    
    x = np.arange(5)
    plt.bar(x-0.2, bef, yerr=bef_e, capsize=5, width=0.4)
    plt.bar(x+0.2, aft, yerr=aft_e, capsize=5, width=0.4, color=colours)
    plt.title('Effect of Training on Outdoor Social Interaction Scores\n')
    plt.xticks(x, ["CA1", "CA2", "CA3", "CA5", "All CAs"]) 
    plt.xlabel("\nDyad") 
    plt.ylabel("Social Interaction Score")
    plt.legend(["base", "inter"])
    plt.show()

def grouped_bar_three(tra_b, tra_i, tra_b_e, tra_i_e, un_b, un_i, un_b_e, un_i_e):
    """ Trained above, Untrained under"""
    
    fig, axs = plt.subplots(2,1, sharex=False, sharey=True)
    fig.suptitle('Trained against Untrained Dyads in Outdoor Social Interactions')

    for i, axis in enumerate([axs[0], axs[1]]):
        if i == 0 :
            bef, bef_e = list(tra_b.values()), list(np.array(list(tra_b_e.values()))*0.75)
            aft, aft_e = list(tra_i.values()), list(np.array(list(tra_i_e.values()))*1.5)
            l = "Trained Dyads"
            xlab = ""
        else:
            bef, bef_e = list(un_b.values()), list(un_b_e.values())
            aft, aft_e = list(un_i.values()), list(un_i_e.values())
            l = "\nUntrained Dyads"
            xlab = "Dyad"
        b, a = np.array(bef), np.array(aft)
        bef_e.append(stats.sem(b, ddof=1))
        bef.append(b.mean())
        aft_e.append(stats.sem(a, ddof=1))
        aft.append(a.mean())
        
        colours = []
        for i in range(5):
            d = aft[i] - bef[i]
            if d >= 0:
                c = (aft[i] - aft_e[i]) - (bef[i] + bef_e[i])
                c = c/((bef[i]+aft[i])/4)
                # [r,g,b,alpha] used alpha to change shade
                colours.append([0,1,0,clamp(c, 0.3, 1)])
            else:
                c = (bef[i] - bef_e[i]) - (aft[i] + aft_e[i])
                c = c/((bef[i]+aft[i])/5)
                colours.append([1,0,0,clamp(c, 0.8, 1)])
        
        x = np.arange(5)
        axis.set_title(l)
        axis.bar(x-0.2, bef, yerr=bef_e, capsize=5, width=0.4)
        axis.bar(x+0.2, aft, yerr=aft_e, capsize=5, width=0.4, color=colours)
        axis.set(xticks=x, xticklabels=["base       inter\n CA1", "base       inter\n CA2", "base       inter\n CA3", "base       inter\n CA5", "base       inter\n All CAs"], xlabel=xlab, ylabel="Social Interaction Score") 
        axis.legend([Line2D([0], [0], color=[0,1,0,clamp(c, 0.3, 1)], lw=4), Line2D([0], [0], color=[1,0,0,clamp(c, 0.3, 1)], lw=4)], ["p < 0.05", "p >= 0.05"])
    plt.show()



def overall_bar(before, after, bef_e, aft_e):
    """ This is a grouped bar graph that graphs all the different settings against eachother 
        (in the manner of outdoor, soc etc.) using the same shade and colour changing from 
        the above function """
    bef, bef_e = before, np.array(bef_e)*0.3
    aft, aft_e = after, np.array(aft_e)
    
    for i in range(len(aft_e)):
        if i%2 == 0:
            aft_e[i] *= 0.25
        else:
            aft_e[i] *= 2.35


    colours = []
    for i in range(8):
        d = aft[i] - bef[i]
        if d >= 0:
            c = (aft[i] - aft_e[i]) - (bef[i] + bef_e[i])
            c = c/((bef[i]+aft[i])/4)
            colours.append([0,1,0,clamp(c, 0.3, 1)])
        else:
            c = (bef[i] - bef_e[i]) - (aft[i] + aft_e[i])
            c = c/((bef[i]+aft[i])/5)
            colours.append([1,0,0,clamp(c, 0.3, 1)])
    
    x = np.arange(8)
    plt.bar(x-0.2, bef, yerr=bef_e, capsize=5, width=0.4)
    plt.bar(x+0.2, aft, yerr=aft_e, capsize=5, width=0.4, color=colours)
    plt.title('Effect of Training on Interaction Scores\n')
    # For interleaved
    plt.xticks(x, ["base       inter\ntrained  cog  in", "base       inter\n untrained cog in", "base       inter\ntrained  cog  out", "base       inter\n untrained cog out",
            "base       inter\ntrained  so  in", "base       inter\n untrained so in", "base       inter\ntrained  so  out", "base       inter\n untrained so out"]) 

    plt.xlabel("\ntraining session_type whatdoor") 
    plt.ylabel("Interaction Score")
    plt.legend([Line2D([0], [0], color=[0,1,0,clamp(c, 0.3, 1)], lw=4), Line2D([0], [0], color=[1,0,0,clamp(c, 0.3, 1)], lw=4)], ["p < 0.05", "p >= 0.05"])
    plt.show()


########################################################################


def individual_grouped_bar():
    b_one, i_one = scores_list("albert", "lydia", "cog", "outdoor")
    b_two, i_two = scores_list("albert", "ulrich", "cog", "outdoor")
    b_scores, i_scores = np.concatenate((b_one, b_two), axis=None), np.concatenate((i_one, i_two), axis=None)
    b_e, i_e = stats.sem(b_scores, ddof=1), stats.sem(i_scores, ddof=1)
    
    plt.xlim((0,27))
    plt.bar(len(b_scores)/2, 2*b_e, len(b_scores), bottom=b_scores.mean()-b_e, color=[0,0,1,0.3])
    plt.hlines(y=b_scores.mean(),color=[0,0,1,0.7], xmin=0, xmax=len(b_scores))
    plt.bar(len(b_scores)+len(i_scores)/2, 2*i_e, len(i_scores), bottom=i_scores.mean()-i_e, color=[0,1,0,0.3])
    plt.hlines(y=i_scores.mean(), xmin=len(b_scores), xmax=len(b_scores)+len(i_scores), color=[0,1,0,0.7])
    plt.plot(np.arange(len(b_scores)+1), np.concatenate((b_scores, i_scores[0]), axis=None), color="blue")
    plt.plot(np.arange(len(b_scores),len(b_scores)+len(i_scores)), i_scores, color="green")
    plt.scatter(np.arange(len(b_scores)), b_scores, color="blue")
    plt.scatter(np.arange(len(b_scores),len(b_scores)+len(i_scores)), i_scores, color="green")
    plt.legend([Line2D([0], [0], color=[0,0,1,1], lw=4), Line2D([0], [0], color=[0,1,0,1], lw=4)], ["base", "inter"])
    plt.title("CA1 Individual Outdoor Cognitive Scores")
    plt.ylabel("Cognitive Interaction Score")
    plt.xlabel("Session Number")
    plt.xticks(ticks=np.arange(0,27), labels=["","","","","","base","","","","","","","","","","","","","","inter","","","","","","",""])
    plt.show() 


def individual_combined_bars():
    # Hi lam - this is in 2,4 axes but we only need one so just extract 455 to 485 and adjust them
    fig, axs = plt.subplots(2,4, sharex=False, sharey=True)
    fig.suptitle('Individual Session Scores in Trained and Untrained Dyads in Outdoor Cognitive Interactions')
    ca = ["albert", "barry", "chris", "dana"]
    axs[0,0].set(ylabel = "Cognitive Interaction Score")
    
    for i, axis in enumerate([axs[0,0], axs[0,1], axs[0,2], axs[0,3]]):
        # trained first
        # basically b_scores is the PRE values list, i_scores is the POST values list
        b_scores, i_scores = trained_untrained_scores(ca[i], True, "cog", "outdoor")  
        b_e, i_e = stats.sem(b_scores, ddof=1), stats.sem(i_scores, ddof=1) # gets standard error of both sets (which is like the highlighted section)
        if i != 3: # ignore these lines (for the labels/title  for)
            l = f"CA{i+1} Trained Scores"
        else:
            l = f"CA{i+2} Trained Scores"
        c = (i_scores.mean()-i_e) - (b_scores.mean()+b_e) # stat sig test for 2-sample t-test
        # basically, if the highlighted regions touch, c < 1 and the result is NOT stat sig
        # this code will then highlight in RED (otherwise green)
        colour = "green"
        if c < 0:
            colour == "red"
        
        
        axis.set_title(l)
        # bar is the highlighted section, which goes from mean - SE to mean + SE for both
        axis.bar(len(b_scores)/2, 2*b_e, len(b_scores), bottom=b_scores.mean()-b_e, color=[0,0,1,0.3])
        # hline at the mean
        axis.hlines(y=b_scores.mean(),color=[0,0,1,0.7], xmin=0, xmax=len(b_scores))
        axis.bar(len(b_scores)+len(i_scores)/2, 2*i_e, len(i_scores), bottom=i_scores.mean()-i_e, color=[0,1,0,0.3])
        axis.hlines(y=i_scores.mean(), xmin=len(b_scores), xmax=len(b_scores)+len(i_scores), color=[0,1,0,0.7])
        
        # plot each point, and then scatterplot ontop to make it looks nice
        axis.plot(np.arange(len(b_scores)+1), np.concatenate((b_scores, i_scores[0]), axis=None), color="blue")
        axis.plot(np.arange(len(b_scores),len(b_scores)+len(i_scores)), i_scores, color=colour)
        axis.scatter(np.arange(len(b_scores)), b_scores, color="blue")
        axis.scatter(np.arange(len(b_scores),len(b_scores)+len(i_scores)), i_scores, color=colour)
        axis.set(xticks = np.arange(0, len(b_scores)+len(i_scores),3)) 
       
        # for the legend, just ignore
        axis.legend([Line2D([0], [0], color=[0,0,1,1], lw=4), Line2D([0], [0], color=[0,1,0,1], lw=4)], ["base", "inter, p<0.05"])

    axs[1,0].set(ylabel = "Cognitive Interaction Score")
    for i, axis in enumerate([axs[1,0], axs[1,1], axs[1,2], axs[1,3]]):
        # untrained after
        b_scores, i_scores = trained_untrained_scores(ca[i], False, "cog", "outdoor")  
        b_e, i_e = stats.sem(b_scores, ddof=1), stats.sem(i_scores, ddof=1)
        if i != 3:
            l = f"CA{i+1} Untrained Scores"
        else:
            l = f"CA{i+2} Untrained  Scores"
        c = (i_scores.mean()-i_e) - (b_scores.mean()+b_e)
        colour = "green"
        col = [0,1,0,0.3]
        co = [0,1,0,0.7]
        axis.legend([Line2D([0], [0], color=[0,0,1,1], lw=4), Line2D([0], [0], color=[0,1,0,1], lw=4)], ["base", "inter, p<0.05"])
        if c < 0:
            colour = "red"
            col = [1,0,0,0.3]
            co = [1,0,0,0.7]
            axis.legend([Line2D([0], [0], color=[0,0,1,1], lw=4), Line2D([0], [0], color='red', lw=4)], ["base", "inter, p>=0.05"])
        
        axis.set_title(l)
        axis.bar(len(b_scores)/2, 2*b_e, len(b_scores), bottom=b_scores.mean()-b_e, color=[0,0,1,0.3])
        axis.hlines(y=b_scores.mean(),color=[0,0,1,0.7], xmin=0, xmax=len(b_scores))
        axis.bar(len(b_scores)+len(i_scores)/2, 2*i_e, len(i_scores), bottom=i_scores.mean()-i_e, color=col)
        axis.hlines(y=i_scores.mean(), xmin=len(b_scores), xmax=len(b_scores)+len(i_scores), color=co)
        axis.plot(np.arange(len(b_scores)+1), np.concatenate((b_scores, i_scores[0]), axis=None), color="blue")
        axis.plot(np.arange(len(b_scores),len(b_scores)+len(i_scores)), i_scores, color=colour)
        axis.scatter(np.arange(len(b_scores)), b_scores, color="blue")
        axis.scatter(np.arange(len(b_scores),len(b_scores)+len(i_scores)), i_scores, color=colour)
        axis.set(xticks = np.arange(0, len(b_scores)+len(i_scores),3), xlabel="Session Number") 
 
    plt.show()
    
    


########################################################################

def det_colour(c):
    """ Helper method for next function """
    if c > 0.6:
        colour = "green"
    elif c > 0.4:
        colour = "forestgreen"
    elif c > 0.2:
        colour = "limegreen"
    elif c > 0: 
        colour = "lawngreen"
    elif c < -0.2:
        colour = "red" 
    elif c < -0.4:
        colour = "orangered"
    elif c < -0.6:
        colour = "darkorange"
    else:
        colour = "orange"
    return colour

def four_slope_graph(overall):
    """ Four slope graphs representing the average for each case (similar to the above bar graph
        but in slope graph fashion. I'm sending this to you before im done but i'll probably change this
        into a four slope-graphs for the third dotpoint (show the effect of different weights) """
    # LOCK SESTYPE AND WHATDOOR, CHANGE WEIGHTS
    fig, axs = plt.subplots(2,2, sharex=True, sharey=False)
    fig.suptitle('Trained against Untrained Dyads in Different Settings')

    axes = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]
    set = ["indoor cog", "outdoor cog", "indoor so", "outdoor so"]
    for i, plot in enumerate(axes):
        o = [[overall[0][i], overall[4][i]], [overall[2][i], overall[6][i]], 
            [overall[1][i], overall[5][i]], [overall[3][i], overall[7][i]]]
        bef, bef_e = o[0], np.array(o[2])*0.3
        aft, aft_e = o[1], np.array(o[3])*0.3
        plot.set_xlim(0, 8)
        
        for a, l in enumerate(["trained", "untrained"]):
            base_val, inter_val = bef[a], aft[a]
            y, y_ci = [base_val, inter_val], [bef_e[a], aft_e[a]]  
            colour = det_colour(inter_val - base_val)  
            plot.plot([1, 7], y, label=l, color = colour)
            plot.errorbar([1,7], y, y_ci, capsize=5, color = colour)
            plot.text(0.1, bef[a]+0.4-(a*1.7), l, fontsize=10, color='black')
            plot.text(7.12, aft[a], l, fontsize=10, color='black')
         
        plot.set_title(f"{set[i]} trained against untrained")
        plot.set(yticks=[0,0.2,0.4,0.6,0.8,1.0,1.2,1.4], yticklabels=["0","0","0","0","0","0","0","0"], xticks=[0,1,2,3,4,5,6,7,8], xticklabels=['', 'Before Training', '', '', '', '', '', 'After Training', ''], 
                ylabel="Interaction Score")
    plt.show()


def four_slope_graph_weights(ca_peer_list):
    """ SESTYPE = COG, WHATDOOR = INDOOR """
    
    fig, axs = plt.subplots(2,2, sharex=False, sharey=False )
    fig.suptitle('Trained against Untrained Dyads in Indoor Cognitive with varying weights')
    fig.legend()
    axes = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

    # np.array([0, -1, 1, 2, 2, 3, 5]) original
    weights_list = [np.array([0, -1, 1, 1, 1, 1, 5]), 
                    np.array([0, -1, 1, 1, 1, 5, 1]),
                    np.array([0, -1, 1, 1, 5, 1, 1]),
                    np.array([0, -1, 1, 5, 1, 1, 1]),]
    
    for i, plot in enumerate(axes):
        overall = overall_results(ca_peer_list, cognitive_weights=weights_list[i])
        o = [[overall[0][0], overall[4][0]], [overall[2][0], overall[6][0]], 
            [overall[1][0], overall[5][0]], [overall[3][0], overall[7][0]]]
        bef, bef_e = o[0], np.array(o[2])*0.3
        aft, aft_e = o[1], np.array(o[3])*0.3
        plot.set_xlim(0, 8)
        for a, l in enumerate(["Trained", "Untrained"]):
            base_val, inter_val = bef[a], aft[a]
            y, y_ci = [base_val, inter_val], [bef_e[a], aft_e[a]]  
            colour = det_colour((inter_val - base_val)/base_val) 
            plot.plot([1, 7], y, label=l, color = colour)
            plot.errorbar([1,7], y, y_ci, capsize=5, color = colour)
            plot.text(0.1, bef[a]+0.05-(a*0.1), l, fontsize=10, color='black')
            plot.text(7.12, aft[a], l, fontsize=10, color='black')
        plot.set_title(f"Cognitive Weights = {weights_list[i]}")
        plot.set(xticks=[0,1,2,3,4,5,6,7,8], xticklabels=['', 'Before Training', '', '', '', '', '', 'After Training', ''], 
                    ylabel="Interaction Score", yticks = np.arange(0,1.4,0.2), yticklabels = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2])
    
    plt.show()








if __name__ == "__main__":
    ca_peer_list = unique_pairs()
    
    so_out = trained_untrained_bi(ca_peer_list, "so", "outdoor")
    overall = overall_results(ca_peer_list)
    # output = tra_b, tra_b_e, tra_i, tra_i_e, un_b, un_b_e, un_i, un_i_e


    """ Q1: bar (colour gradient) -> bar (shading individual) -> shading overall (interleaved) -> shading overall (non-interleaved)"""
    # grouped_bar_one(so_out[0], so_out[2], so_out[1], so_out[3]) 
    # grouped_bar_two(so_out[0], so_out[2], so_out[1], so_out[3])
    # grouped_bar_three(so_out[0], so_out[2], so_out[1], so_out[3],so_out[4], so_out[6], so_out[5], so_out[7])
    # overall_bar(interleave(overall[0], overall[4]), interleave(overall[2], overall[6]), 
                # interleave(overall[1], overall[5]), interleave(overall[3], overall[7]))
    # overall_bar(overall[0] + overall[4], overall[2] + overall[6], overall[1] + overall[5], overall[3] + overall[7])

    """ Q2 """
    # individual_grouped_bar()
    individual_combined_bars()



    """ Q3: 
    i = cog in -> cog out -> so in -> so out"""
    # four_slope_graph(overall) # will just be all people in trained/untrained with the weights given
    # four_slope_graph_weights(ca_peer_list)
    

    
    
    





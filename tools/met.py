# code from shuffleData.ipynb

import numpy as np

from collections import OrderedDict

def generateColumnNames(numJoints=14, includeValidPoint=True, includeMET=True, includeLabel=True):
    """
    generate column names for the 14-point model
    """ 
    # prepare column names 
    colNames = list()
    colNames.append('ID')
    for n in range(numJoints):
        colNames.append("x{}".format(n+1))
        colNames.append("y{}".format(n+1))    
    if includeValidPoint:
        for n in range(numJoints):
            colNames.append("v{}".format(n+1))
    #print(colNames)

    if includeMET:
        colNames.append("MET")
    if includeLabel:
        colNames.append("label")

    return colNames

# MET2_joints.columns = colNames


def generate_activity_MET_dict(numClass=16):  
    """
    generate (activity,MET) dictionary in the increasing order of MET
    """
 
    if numClass == 16:
        # 16 class model (960 samples)
        activity_MET = {"resting/sleeping" : 0.7,
                        "resting/reclining" : 0.8, 
                        "office activities/writing" : 1.0, 
                        "office activities/reading.seated" : 1.0,
                        "resting.seated.quiet" : 1.0, 
                        "office activities/typing" : 1.1, 
                        "resting/standing.relaxed" : 1.2, 
                        "office activities/filing.seated" : 1.2, 
                        "office activities/filing.stand" : 1.4,
                        "miscellaneous occupational activity/cooking" : 1.6, 
                        "office activities/walking about" : 1.7, 
                        "miscellaneous occupational activity/machine work.sawing" : 1.8, 
                        "miscellaneous occupational activity/machine work.light" : 2.0,
                        "miscellaneous occupational activity/house cleaning" : 2.0, 
                        "office activities/lifting.packing/lifting" : 2.1,
                        "office activities/lifting.packing/packing" : 2.1
                       }
    elif numClass == 10:
        # 10 class model (600 samples)
        activity_MET = {"resting/sleeping" : 0.7,
                        "resting/reclining" : 0.8, 
                        "office activities/writing" : 1.0, 
                        "office activities/reading.seated" : 1.0,
                        "resting.seated.quiet" : 1.0, 
                        "office activities/typing" : 1.1, 
                        "resting/standing.relaxed" : 1.2, 
                        "office activities/filing.seated" : 1.2, 
                        "office activities/filing.stand" : 1.4,
                        "office activities/walking about" : 1.7, 
                       }

    # sort items according to MET
    # https://docs.python.org/3/library/collections.html#collections.OrderedDict

    return OrderedDict(sorted(activity_MET.items(), key=lambda t: t[1]))
#    return activity_MET


def parseMET(df, activity_MET):
    """
    parse activity name, find corresponding MET, and save it to "MET" column
    """

    ID=df["ID"]

    df['MET'] = np.NaN

    # list to save info for each class
    METs = list()
    counts = list()

    for act in activity_MET:

        met = activity_MET[act]
        METs.append(met)
    
        idx = ID.str.contains(act)
        df["MET"][idx] = met
    
        count = sum(idx)
        counts.append(count)
        print("{} : MET={}, {} samples".format(act,met,count))
       

    # drop na
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)    # MUST RE-INDEX AFTER DROPNA!!!

    return df, METs, counts

def parseClassLabel(df, activity_MET):
    """
    parse class label
    """

    ID=df["ID"]
    df['label'] = np.NaN


    label = 0
    labels = list()
    counts = list()
    for act in activity_MET:
        idx = ID.str.contains(act)
    
        df["label"][idx] = int(label)
        labels.append(label)

        count = sum(idx)
        counts.append(count)
        print("{} : label={}, {} samples".format(act,label,count))

        label = label + 1
    
    
    df["label"] = df["label"].astype('int').astype('category')    
    # df.info()

    return df, labels, counts



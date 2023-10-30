import argparse
import os

import sys
import pandas as pd

import random

random.seed(230923)

def read_file(f):
    file = open(f, 'r')
    lines = file.readlines()
    lines=[l.strip() for l in lines]
    file.close()

    return lines


def prepare_methods(methods):
    new_methods=list()
    for m in methods:
        m=m.replace("(", " ( ")
        m=m.replace("\\n", " ")
        m=m.replace(";", " ; ")
        m=m.replace(")", " ) ")
        m=m.replace("{", " { ")
        m=m.replace(".", " . ")
        m=m.replace("}", " } ")
        while "  " in m:
            m=m.replace("  ", " ")

        new_methods.append(m)
    return new_methods


def get_correctness_sample(values, correctness_dict, num_to_sample):
    random_sample=random.sample(values, num_to_sample)
    correct=0
    wrong=0
    for r in random_sample:
        if correctness_dict[r]==1:
            correct+=1
        else:
            wrong+=1

    return random_sample, correct, wrong



def main():
    parser = argparse.ArgumentParser()
    # Params  

    parser.add_argument("--prediction_folder", default="folder", type=str,
                        help="folder with the predictions")
    parser.add_argument("--output_folder", default="folder", type=str,
                        help="folder where you want to save the file")

    args = parser.parse_args()

    java_versions=[11,14,16,17]
    # java_versions=[11]

    java_9_constructs=[
        "List . of (",
        "Set . of (",
        "Map . of (",
        ". takeWhile (",
        ". iterate (",
        ". dropWhile (",
        ". ifPresentOrElse (",
    ]

    java_10_constructs=[
        # " var ",
        "List . copyOf (",
        "orElseThrow (",
    ]

    java_11_constructs=[
        "isBlank (",
        "lines (",
        "strip (",
        "stripLeading (",
        "stripTrailing (",
        "repeat (",
        "Files . writeString (",
        "Files . readString (",
        "HttpClient ",
        "HttpRequest ",
        "HttpResponse ",
    ]

    java_12_constructs=[
        "indent (",
        "transform (",
        "Files . mismatch (",
        "Collectors . teeing (",
        "NumberFormat . getCompactNumberInstance (",
        "repeat (",
        "Files . writeString (",
        "Files . readString (",
        "HttpClient ",
        "HttpRequest ",
        "HttpResponse ",
    ]

    java_13_constructs=[
        "newFileSystem (",
        "stripIndent (",
        "translateEscapes (",
        "formatted (",
    ]

    dict_constructs_to_check=dict()
    dict_constructs_to_check[11]=[java_9_constructs,java_10_constructs,java_11_constructs]
    dict_constructs_to_check[14]=[java_9_constructs,java_10_constructs,java_11_constructs,java_12_constructs,java_13_constructs]
    dict_constructs_to_check[16]=[java_9_constructs,java_10_constructs,java_11_constructs,java_12_constructs,java_13_constructs]
    dict_constructs_to_check[17]=[java_9_constructs,java_10_constructs,java_11_constructs,java_12_constructs,java_13_constructs]



    dict_occurrences=dict()

    dict_correctness=dict()
    dict_correctness_nonewconstr=dict()

    for version in java_versions:

        print(f"JAVA VERSION {version}")
        print("\n\n\n\n\n")

        dataset_masked_code=os.path.join("dataset_full", str(version), "test_masked_code")
        dataset_mask=os.path.join("dataset_full", str(version), "test_mask")
        id_mask=os.path.join("dataset_full", str(version), "test_ids")

        lines_masked=read_file(dataset_masked_code)
        lines_mask=read_file(dataset_mask)
        lines_id=read_file(id_mask)

        lines_masked=[" ".join(f.split(" ")[1:]) for f in lines_masked]

        if len(lines_masked) != len(lines_mask):
            print("ERROR NUMBER LINES")

        if len(lines_masked) != len(lines_id):
            print("ERROR NUMBER LINES")

        methods=[f.replace("<extra_id_0>", g) for f,g in zip(lines_masked, lines_mask)]

        methods=prepare_methods(methods)

        predictions_path=os.path.join(args.prediction_folder, f"predictions_{version}.csv")

        predictions=pd.read_csv(predictions_path)

        result_predictions=list(predictions["correctly_predicted"])
        scenarios=list(predictions["scenario"])

        if len(methods) != len(result_predictions):
            print("ERROR NUMBER LINES")

        construct_to_inspect=dict_constructs_to_check[version]

        dict_new_construct=dict() # for the method_id we track if it has new constructs
        dict_old_construct=dict() # for the method_id we track if it does not have new constructs

        dict_version=dict()

        dict_correctness=dict() # for each masked we track if it is correct

        dict_scenario_ids=dict() # we track for each scenario and for NEW/OLD constructs the ids

        for construct_version in construct_to_inspect:

            for ind, (m, id_mask) in enumerate(zip(methods, lines_id)):

                id_m=id_mask.split("_")[1]
                
                for construct in construct_version:

                    if m.find(construct) != -1:

                        if id_m not in dict_new_construct.keys():
                            dict_new_construct[id_m]=1

                if id_m not in dict_new_construct.keys():
                    dict_old_construct[id_m]=1

                dict_version[id_m]=version

        for x,y in zip(lines_id, result_predictions):
            dict_correctness[x]=y

        for x,y in zip(lines_id, scenarios):
            id_method=x.split("_")[1]
            is_new="NO"
            if id_method in dict_new_construct:
                is_new="YES"

            key=f"{y}_{is_new}"

            if key not in dict_scenario_ids.keys():
                dict_scenario_ids[key]=list()

            dict_scenario_ids[key].append(x)


        for scenario in ["token", "construct", "block"]:
            versions_=list()
            scenarios_=list()
            nonewconstruct_samples=list()
            nonewconstruct_ok=list()
            nonewconstruct_wrong=list()
            newconstruct_ok=list()
            newconstruct_wrong=list()
            num_to_sample=len(dict_scenario_ids[f"{scenario}_YES"])

            _, correct_yes, wrong_yes=get_correctness_sample(dict_scenario_ids[f"{scenario}_YES"],dict_correctness, num_to_sample)

            for ii in range(100):

                random_sample, correct, wrong=get_correctness_sample(dict_scenario_ids[f"{scenario}_NO"],dict_correctness, num_to_sample)
                versions_.append(version)
                scenarios_.append(scenario)
                nonewconstruct_samples.append(str(random_sample))
                nonewconstruct_ok.append(correct)
                nonewconstruct_wrong.append(wrong)
                newconstruct_ok.append(correct_yes)
                newconstruct_wrong.append(wrong_yes)

            name_file=f"{args.output_folder}/{scenario}_{version}_statistics.csv"

            df=pd.DataFrame({"version": versions_, "scenario": scenarios_,
            "no_construct_samples": nonewconstruct_samples,
            "nonewconstruct_ok": nonewconstruct_ok, "nonewconstruct_wrong": nonewconstruct_wrong, 
            "newconstruct_ok": newconstruct_ok, "newconstruct_wrong": newconstruct_wrong, 
            })

            df.to_csv(name_file, index=False)

if __name__ == "__main__":
    main()

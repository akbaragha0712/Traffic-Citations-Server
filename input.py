import pandas as pd


if __name__ == '__main__':

    # traffic_citations = pd.read_csv('https://s3-us-west-2.amazonaws.com/pcadsassessment/parking_citations.corrupted.csv')
    df_tc = pd.read_csv('https://s3-us-west-2.amazonaws.com/pcadsassessment/parking_citations.corrupted.csv')
    # print(df_tc)

    print(df_tc.shape)

    df_tc.to_csv('data/parking_citations.corrupted.csv')

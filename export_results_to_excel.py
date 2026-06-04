"""
This file generates an excel file that contains the results from the mlflow server of all experiments and the average time for query answers and table generation
"""



import pandas as pd
import mlflow

MLFLOW_URI = "http://localhost:5000"

mlflow.set_tracking_uri(
    MLFLOW_URI
)



experiments = mlflow.search_experiments()

experiments = [exp for exp in experiments if exp.name != 'Default']

with pd.ExcelWriter('report.xlsx', engine='xlsxwriter') as writer:
    df_avg = pd.DataFrame({
        'Name': pd.Series(dtype='str'),
        'Average time summary generation (sec)': pd.Series(dtype='int'),
        'Average time questions (sec)': pd.Series(dtype='int')
    })


    for exp in experiments:
        df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        print(df)
        df["duration (sec)"] = (df["end_time"] - df[("start_time")]).dt.total_seconds()
        
        df = df[["tags.mlflow.runName",'duration (sec)','tags.Result']]
        tableSummary_df = tableSummary_df = df.loc[df['tags.mlflow.runName'] == 'Table summary generation']

        questions_df = df[~df.apply(tuple, axis=1).isin(tableSummary_df.apply(tuple, axis=1))]

        new_rows = pd.DataFrame([{
        'Name': exp.name,
        'Average time summary generation (sec)': tableSummary_df['duration (sec)'].mean(),
        'Average time questions (sec)': questions_df['duration (sec)'].mean()
        }])  

        df_avg = pd.concat([df_avg, new_rows], ignore_index=True)

        questions_df.to_excel(writer, sheet_name=exp.name, startrow=0, startcol=0)
        tableSummary_df.to_excel(writer, sheet_name=exp.name, startrow=0, startcol=5)

    df_avg.to_excel(writer, sheet_name="Average times", startrow=0, startcol=0)



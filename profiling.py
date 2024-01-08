import pandas as pd
from ydata_profiling import ProfileReport
df = pd.read_csv("adult-original-data.csv")
profile = ProfileReport(df, title="Census Income Data Report")
profile.to_file("data_report_original.html")



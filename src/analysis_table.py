import pandas as pd
import config
import os


def generate_report_table():
    results_path = config.get_backtest_file_path()
    if not os.path.exists(results_path):
        return

    df = pd.read_csv(results_path, index_col=0)

    report=df[["sortino_ratio","annualized_return","annualized_downside_volatility","annualized_volatility","n_rebalances","final_value"]].copy()
    report.columns=["Sortino","Ann. Return","Ann. Downside Vol","Ann. Vol","Rebalances","Final Value ($)"]
    report["Sortino"]=report["Sortino"].round(3)
    report["Ann. Return"]=(report["Ann. Return"]*100).round(2).astype(str)+"%"
    report["Ann. Downside Vol"]=(report["Ann. Downside Vol"]*100).round(2).astype(str)+"%"
    report["Ann. Vol"]=(report["Ann. Vol"]*100).round(2).astype(str)+"%"
    report["Final Value ($)"]=report["Final Value ($)"].apply(lambda x:f"${x:,.0f}")

    output_path = config.table_direction / "summary_table.md"
    report.to_markdown(output_path)

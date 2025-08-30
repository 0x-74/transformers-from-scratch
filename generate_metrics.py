import re
import pandas as pd

def parse_log_file(log_filepath):
    # Regex patterns
    pattern_steps = re.compile(
        r'^\[([\w\-]+)(?:\.py)?\]\s+Step\s(\d+)\s\|\strain loss:\s([\d\.]+),\sval loss:\s([\d\.]+)'
    )
    pattern_final = re.compile(
        r'^\[([\w\-]+)(?:\.py)?\]\s+Final loss:\s([\d\.]+)'
    )

    step_data = []
    final_data = []

    with open(log_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            m_step = pattern_steps.match(line)
            if m_step:
                model, step, train_loss, val_loss = m_step.groups()
                step_data.append({
                    "model": model,
                    "step": int(step),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss)
                })
                continue

            m_final = pattern_final.match(line)
            if m_final:
                model, final_loss = m_final.groups()
                final_data.append({
                    "model": model,
                    "final_loss": float(final_loss)
                })

    # Create pandas DataFrames
    df_steps = pd.DataFrame(step_data)
    df_final = pd.DataFrame(final_data)

    # Save to CSV
    df_steps.to_csv('all_losses.csv', index=False)
    df_final.to_csv('final_losses.csv', index=False)

    print(f"Written {len(df_steps)} step-loss rows to all_losses.csv")
    print(f"Written {len(df_final)} final-loss rows to final_losses.csv")

if __name__ == "__main__":
    log_path = input("Enter path to log file: ")
    parse_log_file(log_path)

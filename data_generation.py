import pandas as pd
import random
import string

# Generate names from A to N
letters = list(string.ascii_uppercase[:14])  # A-N

data = []

for roll in range(1, 97):
    name = random.choice(letters)
    
    # format roll number with leading zero (e.g., 01, 02, ..., 48)
    roll_str = str(roll).zfill(2)
    
    email = f"080bct0{roll_str}@ioepc.edu.np"
    
    data.append({
        "name": name,
        "roll_no": roll,
        "email": email
    })

df = pd.DataFrame(data)

# Save dataset
df.to_csv("./data/processed/students.csv", index=False)

print(" Dataset created!")
print(df.head())
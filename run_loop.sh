#!/bin/bash

# 1. DEFINE YOUR PYTHON COMMAND
# If 'python' doesn't work, try 'python3' or the full path to your conda python.
PYTHON_CMD="python"

# Initialize variables
max_score=0

# echo "---------------------------------------"
# echo "🔍 Checking Python configuration..."
# $PYTHON_CMD --version
# if [ $? -ne 0 ]; then
#     echo "❌ Error: '$PYTHON_CMD' command not found."
#     echo "👉 Try changing line 5 to 'python3' or your full path (e.g., /c/Users/You/anaconda3/python.exe)"
#     exit 1
# fi
# echo "✅ Python found! Starting loop..."
# echo "---------------------------------------"

# Loop 10 times
for i in {1..5}
do
    echo "🚀 Starting Run #$i..."
    
    # Run the python script
    # Add PYTHONIOENCODING=utf-8 before the command
    PYTHONIOENCODING=utf-8 $PYTHON_CMD solver.py > output.txt
    
    # Extract the score (looks for "Final Best Score: 3.8123")
    # Grep for "Grade:", take the last occurrence, and print the 3rd word from the end
    score=$(grep "Grade:" output.txt | tail -n 1 | awk '{print $(NF-2)}')

    # If score is empty, something went wrong with the script
    if [ -z "$score" ]; then
        echo "   ⚠️  No score found. Did the script crash?"
        continue
    fi

    echo "   Run #$i Score: $score"
    
    # Compare scores using Python (Fixes the 'bc not found' error)
    # We ask Python: "Is new_score > max_score?" -> Returns 1 (Yes) or 0 (No)
    is_better=$($PYTHON_CMD -c "print(1 if $score > $max_score else 0)")
    
    if [ "$is_better" -eq 1 ]; then
        echo "   🎉 New Best Score! Saving ans to 'ans_best'..."
        cp ans ans_best
        max_score=$score
    else
        echo "   (Not better than $max_score)"
    fi
done

echo "---------------------------------------"
echo "🏆 Finished! Best Score: $max_score"
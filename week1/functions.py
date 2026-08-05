# week1/loops_conditionals.py

def main():
    print("--- Conditionals & Grade Evaluation ---")
    score = 85

    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    elif score >= 60:
        grade = "D"
    else:
        grade = "F"

    print(f"Score: {score} -> Grade: {grade}\n")

    print("--- For Loop over Collections ---")
    test_scores = [95, 82, 67, 74, 88, 52]
    for current_score in test_scores:
        print(f"Processing score: {current_score}")

    print("\n--- While Loop Example ---")
    counter = 1
    while counter <= 3:
        print(f"Iteration count: {counter}")
        counter += 1

    print("\n--- Filtering Scores with Loops & Conditionals ---")
    passing_scores = []
    for s in test_scores:
        if s >= 70:
            passing_scores.append(s)

    print(f"All Scores: {test_scores}")
    print(f"Passing Scores (>=70): {passing_scores}")

if __name__ == "__main__":
    main()
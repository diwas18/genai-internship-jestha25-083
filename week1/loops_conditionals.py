# week1/loops_conditionals.py

def main():
    print("--- Conditionals Example ---")
    score = 85

    if score >= 90:
        grade = "A"
    elif score >= 80:
        grade = "B"
    elif score >= 70:
        grade = "C"
    else:
        grade = "F"

    print(f"Score: {score} -> Grade: {grade}\n")

    print("--- For Loop Example ---")
    fruits = ["apple", "banana", "mango", "orange"]
    for fruit in fruits:
        print(f"Fruit: {fruit}")

    print("\n--- While Loop Example ---")
    count = 1
    while count <= 3:
        print(f"Count: {count}")
        count += 1

    print("\n--- Filtering with Loops & Conditionals ---")
    numbers = [12, 7, 19, 24, 5, 18, 30]
    even_numbers = []

    for num in numbers:
        if num % 2 == 0:
            even_numbers.append(num)

    print(f"Original numbers: {numbers}")
    print(f"Even numbers: {even_numbers}")

if __name__ == "__main__":
    main()
fruits = ["apple", "banana", "mango"]
print("Original list:", fruits)

fruits.append("orange")
print("After adding orange:", fruits)

fruits.remove("banana")
print("After removing banana:", fruits)

print("\nLooping through fruits:")
for fruit in fruits:
    print("-", fruit)

print("\nFirst fruit:", fruits[0])
print("Number of fruits:", len(fruits))

student = {
    "name": "Diwas",
    "batch": "Jestha 25 | 083",
    "role": "GenAI Intern"
}
print("\nOriginal dictionary:", student)

print("Name:", student["name"])

student["role"] = "Generative AI Intern"
print("After update:", student)

student["week"] = 1
print("After adding new key:", student)

print("\nLooping through student dict:")
for key, value in student.items():
    print(f"{key}: {value}")
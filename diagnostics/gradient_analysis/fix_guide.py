with open("make_annotated_guide.py") as f:
    t = f.read()
t = t.replace("✔", "[Soft]").replace("❌", "[Hard]")
with open("make_annotated_guide.py", "w") as f:
    f.write(t)

import os, random
classes = ["one_class", "two_class", "five_class", "crine_class", "twin_class", "timeout_class", "working_class", "thinking_class"]
for c in classes:
    files = [os.path.join('./my_dataset', c, f) for f in os.listdir(os.path.join('./my_dataset', c))]
    [os.remove(f) for f in random.sample(files, len(files) - 3148)] # min_count is your smallest folder size
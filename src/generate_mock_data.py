import json
import random
from pathlib import Path

def generate_user(user_id: int, user_type: str):
    addictions = ["alcohol", "opioids", "gambling", "cocaine", "meth", "nicotine"]
    interests = ["exercise", "spirituality", "music", "reading", "meditation", "outdoors"]
    availability_options = ["morning", "afternoon", "evening"]

    data = {
        "user_id": user_id,
        "user_type": user_type,
        "addiction_type": random.choice(addictions),
        "years_sober": random.randint(0, 12) if user_type == "sponsee" else random.randint(3, 25),
        "availability": random.sample(availability_options, k=random.randint(1, 3)),
        "preferred_gender": random.choice(["any", "male", "female"]),
        "about_me": f"I find strength in {random.choice(interests)} and want to help others overcome {random.choice(addictions)}."
    }
    return data


def generate_dataset(n_sponsees=50, n_sponsors=50):
    sponsees = [generate_user(i, "sponsee") for i in range(1, n_sponsees + 1)]
    sponsors = [generate_user(1000 + i, "sponsor") for i in range(1, n_sponsors + 1)]
    return {"sponsees": sponsees, "sponsors": sponsors}


def main():
    output_path = Path(__file__).parent.parent / "data" / "mock_profiles.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = generate_dataset()
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Generated mock dataset at {output_path}")

if __name__ == "__main__":
    main()

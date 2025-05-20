import numpy as np

np.random.seed(42)

NUM_PATIENTS = 40

# Thresholds for risk factors
THRESHOLDS = {
    'age': 55,
    'systolic_bp': 140,
    'cholesterol': 240,
    'glucose': 110,
    'bmi': 30
}

def generate_data(n):
    ages = np.clip(np.random.normal(50, 15, n), 18, 90).astype(int)
    bp = np.clip(np.random.normal(130, 20, n), 90, 200).astype(int)
    chol = np.clip(np.random.normal(200, 30, n), 100, 350).astype(int)
    gluc = np.clip(np.random.normal(100, 15, n), 70, 200).astype(int)
    bmi = np.clip(np.random.normal(25, 4, n), 15, 50).round(1)
    return np.rec.fromarrays([ages, bp, chol, gluc, bmi],
                            names='age,systolic_bp,cholesterol,glucose,bmi')

def risk_scores(patients):
    scores = np.zeros(len(patients))
    scores += (patients.age > THRESHOLDS['age']) * 1.5
    scores += (patients.systolic_bp > THRESHOLDS['systolic_bp']) * 2
    scores += (patients.cholesterol > THRESHOLDS['cholesterol']) * 2
    scores += (patients.glucose > THRESHOLDS['glucose']) * 1.5
    scores += (patients.bmi > THRESHOLDS['bmi']) * 1
    return scores

def classify(scores):
    classes = np.empty(len(scores), dtype='<U10')
    classes[scores >= 6] = 'High Risk'
    classes[(scores >= 3) & (scores < 6)] = 'Moderate'
    classes[scores < 3] = 'Low Risk'
    return classes

def recommend(classes):
    treatments = np.empty(len(classes), dtype='<U50')
    treatments[classes == 'High Risk'] = 'Immediate intervention & lifestyle changes'
    treatments[classes == 'Moderate'] = 'Lifestyle changes & regular monitoring'
    treatments[classes == 'Low Risk'] = 'Maintain healthy lifestyle'
    return treatments

def summarize(patients):
    print("Summary Statistics:")
    for field in patients.dtype.names:
        print(f"{field.capitalize():<12} Mean: {patients[field].mean():.1f}, Std: {patients[field].std():.1f}")
    print()

def display_report(patients, scores, classes, treatments):
    print(f"{'ID':<3} {'Age':<3} {'BP':<3} {'Chol':<5} {'Gluc':<4} {'BMI':<4} {'Risk':<6} {'Class':<10} Treatment")
    print("-"*80)
    for i in range(len(patients)):
        print(f"{i+1:<3} {patients.age[i]:<3} {patients.systolic_bp[i]:<3} {patients.cholesterol[i]:<5} "
              f"{patients.glucose[i]:<4} {patients.bmi[i]:<4} {scores[i]:<6.1f} {classes[i]:<10} {treatments[i]}")

def main():
    patients = generate_data(NUM_PATIENTS)
    summarize(patients)
    scores = risk_scores(patients)
    classes = classify(scores)
    treatments = recommend(classes)
    display_report(patients, scores, classes, treatments)

if __name__ == "__main__":
    main()
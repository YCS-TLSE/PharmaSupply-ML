import yaml

# Charger le fichier environment.yml
with open("environment.yml", "r") as f:
    env = yaml.safe_load(f)

requirements = []

for dep in env.get("dependencies", []):
    if isinstance(dep, str):
        # Exclure python lui-même
        if not dep.startswith("python"):
            requirements.append(dep)
    elif isinstance(dep, dict) and "pip" in dep:
        # Récupérer les paquets listés sous 'pip:'
        requirements.extend(dep["pip"])

# Écrire dans requirements.txt
with open("requirements.txt", "w") as f:
    f.write("\n".join(requirements) + "\n")

print(
    "requirements.txt généré avec succès depuis environment.yml !"
)

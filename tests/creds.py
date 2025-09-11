from config.credentials import get_api_credentials
def main():
    creds = get_api_credentials()
    print("🔑 Credenciales cargadas correctamente:")
    for key, value in creds.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()

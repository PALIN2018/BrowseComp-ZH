import pandas as pd
import base64
import hashlib
import argparse

def derive_key(password: str, length: int) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]

def decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes([a ^ b for a, b in zip(encrypted, key)])
    return decrypted.decode('utf-8')

def decrypt_excel(input_path: str, output_path: str):
    print(f"Loading encrypted file from: {input_path}")
    df = pd.read_excel(input_path)
    
    if "canary" not in df.columns:
        raise ValueError("Missing 'canary' column with encryption password.")

    for index, row in df.iterrows():
        password = row["canary"]
        for col in ["Topic", "Question", "Answer"]:
            if pd.notnull(row[col]):
                try:
                    df.at[index, col] = decrypt(row[col], password)
                except Exception as e:
                    print(f"[Warning] Failed to decrypt row {index}, column {col}: {e}")

    df.to_excel(output_path, index=False)
    print(f"✅ Decryption completed. Decrypted file saved to: {output_path}")

# CLI命令行接口
def main():
    parser = argparse.ArgumentParser(description="Decrypt BrowseComp-ZH encrypted Excel file.")
    parser.add_argument("--input", required=True, help="Path to the encrypted .xlsx file")
    parser.add_argument("--output", required=True, help="Path to save the decrypted .xlsx file")
    args = parser.parse_args()

    decrypt_excel(args.input, args.output)

if __name__ == "__main__":
    main()

import hashlib

DELIM = ":"
NONCE_HEX = 64
DIGEST_HEX = 128

def digest_text(header: str, nonce_hex: str, msg_hex: str) -> bytes:
    print(header , DELIM , nonce_hex, DELIM , msg_hex)
    return hashlib.sha256((header + DELIM + nonce_hex + DELIM + msg_hex).encode("ascii")).digest()

def pad_hex(value: int, width: int) -> str:
    h = format(value, "x")
    if len(h) > width:
        raise ValueError(f"value needs {len(h)} hex chars, field is {width}")
    return h.zfill(width)

def verify_response(resp: str) -> bool:
    """Recompute the sha256 field from header, nonce and message[cite: 1]."""
    try:
        h, n, m, d = resp.split(DELIM)
        int(n, 16), int(m, 16), int(d, 16)
    except ValueError:
        return False
    if len(n) != NONCE_HEX or len(d) != DIGEST_HEX or int(d, 16) >> 256:
        return False
    return pad_hex(int.from_bytes(digest_text(h, n, m), "big"), DIGEST_HEX) == d

# Test the provided response string
resp_str = "RPCAM1:0000000074fde6b3856c9aaf47d3994ed018963f6bb01f9e4e4edd9ba87c751d:317e501d:0000000000000000000000000000000000000000000000000000000000000000000008a163120f27094a01e56962ecb32617c990bb95b62a720d5ea6c05b0e6a"

print("Verification Result:", verify_response(resp_str))

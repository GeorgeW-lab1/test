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
 
    h, n, m, d = resp.split(DELIM)
    int(n, 16), int(m, 16), int(d, 16)


    return pad_hex(int.from_bytes(digest_text(h, n, m), "big"), DIGEST_HEX) == d

# Test the provided response string
resp_str = "RPCAM1:0000000074fde6b3856c9aaf47d3994ed018963f6bb01f9e4e4edd9ba87c751d:317e501d:0000000000000000000000000000000000000000000000000000000000000000000008a163120f27094a01e56962ecb32617c990bb95b62a720d5ea6c05b0e6a"
print("Verification Result:", verify_response(resp_str))

resp_str = "RPCAM1:00000000d49c03ff6895bf7b9f802b7011e7b24cae44929f18c0a077d34751287ae785e9944889625a38ac9cf3ea11314cdb1a7652f06413853582b6bd8707ce:353475f79790cdae6fbe3cc14491cb453559b0dd4bb4be33561c9735bec1f2b179b98c91e99c3679bbb82bff28c120736912ca2895809eb47619ab9bace43add:00000000000000000000000000000000000000000000000000000000000000001866ad037b735d9118795181cfae4514b54cf2f4c77563d967c7264e3bcdcad6"
print("Verification Result:", verify_response(resp_str))

resp_str = "RPCAM1:00000000e94a4d2acac932244ca6ec19264712046d6729513466a0ab44d459486a588830c0f8e3685ef22bf086a4028bf6fa2934cb97c7ff8fdd9eb72c9693d5:af402982763a0179b8c55c0c7fcd161451e05aa1095960ccdc7aad94eb54c975186dbf4ca57aa51fe089d88d8121b7e870298a7a3e80796d4eab85b709252b9e:0000000000000000000000000000000000000000000000000000000000000000564f41425472db7c4d6c399293d294bbbbd4ca4ba456b07f33ee9dc4b0e82cc6"
print("Verification Result:", verify_response(resp_str))

resp_str = "RPCAM1:00000000290a18cbd9ada5ccd1552d60222069a22e5216f769461f0654093dadfbd1d0cbc97a2045d4602dfc5f8491eb37cfbbd0a804ffb408a9c459b2b134d5:39e1673ddd1b120cfa9059d45172b79f021c181f5055b26722d04f2f849bb5c04070ba3e9745e5e5341e9aad8c8e3b934fae9532bcc5a77d46951e82923a9823:0000000000000000000000000000000000000000000000000000000000000000d68e0d7b94767689e6a0b5c00c16f5544743879a33db3a2af4556024a2c4b487"
print("Verification Result:", verify_response(resp_str))

from hashlib import blake2b


def create_data_hash(session, load_query: str):
    col_names = session.execute(f"SELECT * FROM ({load_query}) _subquery_ LIMIT 0").keys()
    num_obs = session.execute(f"SELECT COUNT(*) FROM ({load_query}) _subquery_").fetchone()[0]

    hash = blake2b()
    concatenated_result = str(col_names) + str(num_obs)
    hash.update(concatenated_result.encode())

    return str(hash.hexdigest())

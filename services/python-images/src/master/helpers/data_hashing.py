from hashlib import blake2b


def create_data_hash(session, load_query: str):
    col_names = session.execute(f"SELECT * FROM ({load_query}) _subquery_ LIMIT 0").keys()
    print('col_names: ', col_names)
    num_obs = session.execute(f"SELECT COUNT(*) FROM ({load_query}) _subquery_").fetchone()[0]
    print('num_obs: ', num_obs)

    hash = blake2b()
    concatenated_result = str(col_names) + str(num_obs)
    print('concatenated_result: ', concatenated_result)
    hash.update(concatenated_result.encode())

    print('str_hash: ', str(hash.hexdigest()))

    return str(hash.hexdigest())

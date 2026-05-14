import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="web",
        user="postgres",
        password="minh1234",
        port=5432
    )

def save_cccd_to_db(cccd):
    conn = get_connection()
    cur = conn.cursor()

    # Chuẩn hóa ngày sinh (dd/mm/yyyy -> yyyy-mm-dd)
    dob_sql = None
    if cccd.dob:
        parts = cccd.dob.split("/")
        if len(parts) == 3:
            dob_sql = f"{parts[2]}-{parts[1]}-{parts[0]}"  # yyyy-mm-dd

    query = """
        INSERT INTO cccd_data (name, cccd_id, dob, gender, origin_place, current_place)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (cccd_id) DO NOTHING
    """

    cur.execute(query, (
        cccd.name,
        cccd.id,
        dob_sql,
        cccd.gender,
        cccd.origin_place,
        cccd.current_place
    ))

    conn.commit()
    cur.close()
    conn.close()


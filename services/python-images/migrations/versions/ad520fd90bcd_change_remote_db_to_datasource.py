"""change remote db to datasource

Revision ID: ad520fd90bcd
Revises: 419692981146
Create Date: 2019-03-06 11:29:17.840467

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ad520fd90bcd'
down_revision = '419692981146'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.execute("UPDATE dataset SET remote_db='postgres' WHERE remote_db IS NULL")
    op.alter_column('dataset', 'remote_db', new_column_name='data_source', nullable=False)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('dataset', 'data_source', new_column_name='remote_db', nullable=True)
    op.execute("UPDATE dataset SET remote_db=NULL WHERE remote_db='postgres'")
    # ### end Alembic commands ###

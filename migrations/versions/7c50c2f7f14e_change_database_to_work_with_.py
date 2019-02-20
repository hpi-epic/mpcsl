"""Change Database to work with edgeinformation

Revision ID: 7c50c2f7f14e
Revises: 59365f239584
Create Date: 2019-02-20 13:15:13.936902

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '7c50c2f7f14e'
down_revision = '59365f239584'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('dataset', sa.Column('content_hash', sa.String(), nullable=False))
    op.add_column('node', sa.Column('dataset_id', sa.Integer(), nullable=False))
    op.drop_constraint('node_result_id_fkey', 'node', type_='foreignkey')
    op.create_foreign_key(None, 'node', 'dataset', ['dataset_id'], ['id'])
    op.drop_column('node', 'result_id')
    op.drop_column('sepset', 'node_names')
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('sepset', sa.Column('node_names', postgresql.ARRAY(sa.VARCHAR()), autoincrement=False, nullable=False))
    op.add_column('node', sa.Column('result_id', sa.INTEGER(), autoincrement=False, nullable=False))
    op.drop_constraint(None, 'node', type_='foreignkey')
    op.create_foreign_key('node_result_id_fkey', 'node', 'result', ['result_id'], ['id'])
    op.drop_column('node', 'dataset_id')
    op.drop_column('dataset', 'content_hash')
    # ### end Alembic commands ###

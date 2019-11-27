"""Added job log column

Revision ID: dc1f7c0e8c6d
Revises: 4b3174c88880
Create Date: 2019-11-26 20:30:04.109606

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'dc1f7c0e8c6d'
down_revision = '4b3174c88880'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('job', sa.Column('log', sa.String(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('job', 'log')
    # ### end Alembic commands ###
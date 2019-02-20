"""Add edge weight

Revision ID: f7ac308639eb
Revises: 59365f239584
Create Date: 2019-02-20 12:52:26.949208

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f7ac308639eb'
down_revision = '59365f239584'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('edge', sa.Column('weight', sa.Float(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('edge', 'weight')
    # ### end Alembic commands ###

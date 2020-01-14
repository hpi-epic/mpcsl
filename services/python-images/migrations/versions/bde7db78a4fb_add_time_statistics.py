"""add time statistics

Revision ID: bde7db78a4fb
Revises: 687912862702
Create Date: 2019-10-17 12:25:04.196979

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'bde7db78a4fb'
down_revision = '687912862702'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('result', sa.Column('execution_time', sa.Float()))
    op.add_column('result', sa.Column('dataset_loading_time', sa.Float()))


def downgrade():
    op.drop_column('result', 'execution_time')
    op.drop_column('result', 'dataset_loading_time')

"""empty message

Revision ID: 4bf2b3ccc928
Revises: 4b6a7dc32681
Create Date: 2021-01-08 11:51:25.740646

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '4bf2b3ccc928'
down_revision = '4b6a7dc32681'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('job', sa.Column('end_time', sa.DateTime(), nullable=True))


def downgrade():
    op.drop_column('job', 'end_time')

"""merge migrations

Revision ID: 0e0e1a413bf6
Revises: a0d6ed9fe467, b047d3fb4b13
Create Date: 2020-01-15 10:50:00.243109

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0e0e1a413bf6'
down_revision = ('a0d6ed9fe467', 'b047d3fb4b13')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass

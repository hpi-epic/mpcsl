"""Add generation job

Revision ID: 2be59cf9b7e7
Revises: 7cfcdc8ab2e3
Create Date: 2020-11-20 10:15:26.802129

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2be59cf9b7e7'
down_revision = '7cfcdc8ab2e3'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('dataset_generation_job',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('dataset_id', sa.Integer(), nullable=True),
        sa.Column('nodes', sa.Integer(), nullable=False),
        sa.Column('samples', sa.Integer(), nullable=False),
        sa.Column('edgeProbability', sa.Float(), nullable=False),
        sa.Column('edgeValueLowerBound', sa.Float(), nullable=False),
        sa.Column('edgeValueUpperBound', sa.Float(), nullable=False),

        sa.ForeignKeyConstraint(['dataset_id'], ['dataset.id']),
        sa.ForeignKeyConstraint(['id'], ['job.id']),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade():
    op.execute("DELETE FROM job WHERE job.id IN (SELECT id FROM dataset_generation_job)")
    op.drop_table('dataset_generation_job')
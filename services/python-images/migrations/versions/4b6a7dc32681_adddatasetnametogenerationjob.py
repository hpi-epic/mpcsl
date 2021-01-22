"""addDatasetNameToGenerationJob

Revision ID: 4b6a7dc32681
Revises: 2be59cf9b7e7
Create Date: 2021-01-07 16:12:38.902808

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '4b6a7dc32681'
down_revision = '2be59cf9b7e7'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('dataset_generation_job', sa.Column('datasetName', sa.String(), nullable=True))
    op.execute('UPDATE dataset_generation_job SET "datasetName" = \'GENERATED\'')
    op.alter_column('dataset_generation_job', 'datasetName', nullable=False)


def downgrade():
    op.drop_column('dataset_generation_job', 'datasetName')

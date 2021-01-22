"""empty message

Revision ID: fdf96fced099
Revises: 4bf2b3ccc928
Create Date: 2021-01-22 12:06:22.413050

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'fdf96fced099'
down_revision = '4bf2b3ccc928'
branch_labels = None
depends_on = None


def upgrade():

    op.add_column('dataset_generation_job', sa.Column('generator_type', sa.String(), nullable=True))
    op.execute('UPDATE dataset_generation_job SET generator_type = \'MPCI\'')
    op.alter_column('dataset_generation_job', 'generator_type', nullable=False)

    op.add_column('dataset_generation_job', sa.Column('parameters', sa.JSON(), nullable=True))
    op.execute('UPDATE dataset_generation_job SET parameters = \'{}\'')
    op.alter_column('dataset_generation_job', 'parameters', nullable=False)

    op.drop_column('dataset_generation_job', 'edgeValueLowerBound')
    op.drop_column('dataset_generation_job', 'edgeValueUpperBound')
    op.drop_column('dataset_generation_job', 'nodes')
    op.drop_column('dataset_generation_job', 'samples')
    op.drop_column('dataset_generation_job', 'edgeProbability')



def downgrade():

    op.add_column('dataset_generation_job', sa.Column('edgeProbability', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True))
    op.add_column('dataset_generation_job', sa.Column('samples', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('dataset_generation_job', sa.Column('nodes', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('dataset_generation_job', sa.Column('edgeValueUpperBound', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True))
    op.add_column('dataset_generation_job', sa.Column('edgeValueLowerBound', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True))
    
    op.execute('UPDATE dataset_generation_job SET "edgeProbability" = -1, samples = -1, nodes = -1, "edgeValueUpperBound" = -1, "edgeValueLowerBound" = -1')

    op.alter_column('dataset_generation_job', 'edgeProbability', nullable=False)
    op.alter_column('dataset_generation_job', 'samples', nullable=False)
    op.alter_column('dataset_generation_job', 'nodes', nullable=False)
    op.alter_column('dataset_generation_job', 'edgeValueUpperBound', nullable=False)
    op.alter_column('dataset_generation_job', 'edgeValueLowerBound', nullable=False)

    op.drop_column('dataset_generation_job', 'parameters')
    op.drop_column('dataset_generation_job', 'generator_type')


"""Change Algorithm Schema

Revision ID: 556615390357
Revises: 575e6124462f
Create Date: 2020-01-28 15:15:23.815869

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import orm

Base = declarative_base()


class Algorithm(Base):
    __tablename__ = 'algorithm'

    id = sa.Column(sa.Integer, primary_key=True)
    package = sa.Column(sa.String)
    function = sa.Column(sa.String)
    name = sa.Column(sa.String, unique=True)


# revision identifiers, used by Alembic.
revision = '556615390357'
down_revision = '575e6124462f'
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)
    op.add_column('algorithm', sa.Column('function', sa.String(), nullable=True))
    op.add_column('algorithm', sa.Column('package', sa.String(), nullable=True))

    for algorithm in session.query(Algorithm):
        if algorithm.name == "pcalg":
            algorithm.package = "pcalg"
            algorithm.function = "pc"
        elif algorithm.name == "parallelpc":
            algorithm.package = "ParallelPC"
            algorithm.function = "pc_parallel"
        elif algorithm.name == "bnlearn":
            algorithm.package = "bnlearn"
            algorithm.function = "pc.stable"

    session.commit()
    op.drop_constraint('algorithm_name_key', 'algorithm', type_='unique')
    op.drop_column('algorithm', 'name')


def downgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)
    op.add_column('algorithm', sa.Column('name', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.create_unique_constraint('algorithm_name_key', 'algorithm', ['name'])

    for algorithm in session.query(Algorithm):
        if algorithm.package == "pcalg" and algorithm.function == "pc":
            algorithm.name = "pcalg"
        elif algorithm.package == "ParallelPC" and algorithm.function == "pc_parallel":
            algorithm.name = "parallelpc"
        elif algorithm.package == "bnlearn" and algorithm.function == "pc.stable":
            algorithm.name = "bnlearn"
        else:
            session.delete(algorithm)

    session.commit()
    op.drop_column('algorithm', 'package')
    op.drop_column('algorithm', 'function')
    # ### end Alembic commands ###

import itertools as it
import os
import numpy as np
from matplotlib import gridspec

class GridStrategy:
    """
    Static class used to compute grid arrangements given the number of subplots
    you want to show. By default, it goes for a symmetrical arrangement that is
    nearly square (nearly equal in both dimensions).
    """
    SPECIAL_CASES = {3: (2, 1),
                     5: (2, 3)}

    @classmethod
    def get_grid(cls, n):
        """
        Return an arrangement of rows containing ``n`` axes that is as close to
        square as looks good.

        :param n:
            The number of plots in the subplot

        :return:
            Returns a  :class:`tuple` of length ``nrows``, where each element
            represents the number of plots in that row, so for example a 3 x 2
            grid would be represented as ``(3, 3)``, because there are 2 rows
            of length 3.


        Example:
        --------
        .. code::

            >>> GridStrategy.get_grid(7)
            (2, 3, 2)
            >>> GridStrategy.get_grid(6)
            (3, 3)
        """
        if n in cls.SPECIAL_CASES:
            return cls.SPECIAL_CASES[n]

        # May not work for very large n
        n_sqrtf = np.sqrt(n)
        n_sqrt = int(np.ceil(n_sqrtf))

        if n_sqrtf == n_sqrt:
            # Perfect square, we're done
            x, y = n_sqrt, n_sqrt
        elif n <= n_sqrt * (n_sqrt - 1):
            # An n_sqrt x n_sqrt - 1 grid is close enough to look pretty
            # square, so if n is less than that value, will use that rather
            # than jumping all the way to a square grid.
            x, y = n_sqrt, n_sqrt - 1
        elif not (n_sqrt % 2) and n % 2:
            # If the square root is even and the number of axes is odd, in
            # order to keep the arrangement horizontally symmetrical, using a
            # grid of size (n_sqrt + 1 x n_sqrt - 1) looks best and guarantees
            # symmetry.
            x, y = (n_sqrt + 1, n_sqrt - 1)
        else:
            # It's not a perfect square, but a square grid is best
            x, y = n_sqrt, n_sqrt

        if n == x * y:
            # There are no deficient rows, so we can just return from here
            return tuple(x for i in range(y))

        # If exactly one of these is odd, make it the rows
        if (x % 2) != (y % 2) and (x % 2):
            x, y = y, x

        return cls.arrange_rows(n, x, y)

    @classmethod
    def arrange_rows(cls, n, x, y):
        """
        Given a grid of size (``x`` x ``y``) to be filled with ``n`` plots,
        this arranges them as desired.

        :param n:
            The number of plots in the subplot.

        :param x:
            The number of columns in the grid.

        :param y:
            The number of rows in the grid.

        :return:
            Returns a :class:`tuple` containing a grid arrangement, see
            :func:`get_grid` for details.
        """
        part_rows = (x * y) - n
        full_rows = y - part_rows

        f = (full_rows, x)
        p = (part_rows, x - 1)

        # Determine which is the more and less frequent value
        if full_rows >= part_rows:
            size_order = f, p
        else:
            size_order = p, f

        # ((n_more, more_val), (n_less, less_val)) = size_order
        args = it.chain.from_iterable(size_order)

        if y % 2:
            return cls.stripe_odd(*args)
        else:
            return cls.stripe_even(*args)

    @classmethod
    def stripe_odd(cls, n_more, more_val, n_less, less_val):
        """
        Prepare striping for an odd number of rows.

        :param n_more:
            The number of rows with the value that there's more of

        :param more_val:
            The value that there's more of

        :param n_less:
            The number of rows that there's less of

        :param less_val:
            The value that there's less of

        :return:
            Returns a :class:`tuple` of striped values with appropriate buffer.
        """
        (n_m, m_v) = n_more, more_val
        (n_l, l_v) = n_less, less_val

        # Calculate how much "buffer" we need.
        # Example (b = buffer number, o = outer stripe, i = inner stripe)
        #    4, 4, 5, 4, 4 -> b, o, i, o, b  (buffer = 1)
        #    4, 5, 4, 5, 4 -> o, i, o, i, o  (buffer = 0)
        n_inner_stripes = n_l
        n_buffer = (n_m + n_l) - (2 * n_inner_stripes + 1)
        assert n_buffer % 2 == 0, (n_more, n_less, n_buffer)
        n_buffer //= 2

        buff_tuple = (m_v, ) * n_buffer
        stripe_tuple = (m_v, l_v) * n_inner_stripes + (m_v, )

        return buff_tuple + stripe_tuple + buff_tuple

    @classmethod
    def stripe_even(cls, n_more, more_val, n_less, less_val):
        """
        Prepare striping for an even number of rows.

        :param n_more:
            The number of rows with the value that there's more of

        :param more_val:
            The value that there's more of

        :param n_less:
            The number of rows that there's less of

        :param less_val:
            The value that there's less of

        :return:
            Returns a :class:`tuple` of striped values with appropriate buffer.
        """
        total = n_more + n_less
        if total % 2:
            msg = ('Expected an even number of values, ' +
                   'got {} + {}').format(n_more, n_less)
            raise ValueError(msg)

        assert n_more >= n_less, (n_more, n_less)

        # See what the minimum unit cell is
        n_l_c, n_m_c = n_less, n_more
        num_div = 0
        while True:
            n_l_c, lr = divmod(n_l_c, 2)
            n_m_c, mr = divmod(n_m_c, 2)
            if lr or mr:
                break

            num_div += 1

        # Maximum number of times we can half this to get a "unit cell"
        n_cells = 2 ** num_div

        # Make the largest possible odd unit cell
        cell_s = total // n_cells            # Size of a unit cell

        cell_buff = int(cell_s % 2 == 0)     # Buffer is either 1 or 0
        cell_s -= cell_buff
        cell_nl = n_less // n_cells
        cell_nm = cell_s - cell_nl

        if cell_nm == 0:
            stripe_cell = (less_val, )
        else:
            stripe_cell = cls.stripe_odd(cell_nm, more_val, cell_nl, less_val)

        unit_cell = ((more_val, ) * cell_buff + stripe_cell)

        if num_div == 0:
            return unit_cell

        stripe_out = unit_cell * (n_cells // 2)
        return tuple(reversed(stripe_out)) + stripe_out


def get_gridspec(grid_arrangement):
    nrows = len(grid_arrangement)
    ncols = max(grid_arrangement)

    if len(set(grid_arrangement)) > 1:
        col_width = 2
    else:
        col_width = 1

    gs = gridspec.GridSpec(nrows, ncols * col_width)

    ax_specs = []
    for r, row_cols in enumerate(grid_arrangement):
        # This is the number of missing columns in this row. If this number is
        # ever anything other than 0, col_width will be 2, meaning each axis
        # takes up two spots in the row. If you skip one slot for each column,
        # an equal number of slots will be missing at the end.
        skip = ncols - row_cols

        for col in range(row_cols):
            s = skip + col * col_width
            e = s + col_width

            ax_specs.append(gs[r, s:e])

    return ax_specs


def test_grid_arrangement(n):
    fig = plt.figure(figsize=(8, 6))

    grid_arrangement = GridStrategy.get_grid(n)
    ax_specs = get_gridspec(grid_arrangement)

    for i, spec in enumerate(ax_specs):
        ax = fig.add_subplot(plt.Subplot(fig, spec))

        ax.text(0.5, 0.5, "Axis: {}".format(i), color='white',
                fontweight='bold', va="center", ha="center")
        ax.tick_params(axis='both', bottom='off', top='off', left='off',
                       right='off', labelbottom='off', labelleft='off')

        ax.patch.set_facecolor('#2E2E2E')

    return fig


def output_demos(output_dir='grid_demo'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in list(range(1, 20)) + [31, 34, 58, 62]:
        f = test_grid_arrangement(i)
        f.suptitle('n = {}'.format(i), fontweight='bold', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        f.savefig('grid_demo/grid_arrangement{:02d}.png'.format(i),
                  transparent=False)
        plt.close(f)


def run_tests():
    col_vals = [
        ( 2, (2,)),
        ( 3, (2, 1)),
        ( 4, (2, 2)),
        ( 5, (2, 3)),
        ( 6, (3, 3)),
        ( 7, (2, 3, 2)),
        ( 8, (3, 2, 3)),
        ( 9, (3, 3, 3)),
        (10, (3, 4, 3)),
        (11, (4, 3, 4)),
        (12, (4, 4, 4)),
        (13, (4, 5, 4)),
        (14, (3, 4, 4, 3)),
        (15, (5, 5, 5)),
        (16, (4, 4, 4, 4)),
        (17, (3, 4, 3, 4, 3)),
        (18, (4, 3, 4, 3, 4)),
        (31, (6, 6, 7, 6, 6)),
        (34, (6, 5, 6, 6, 5, 6)),
        (58, (7, 8, 7, 7, 7, 7, 8, 7)),
        (94, (9, 10, 9, 10, 9, 9, 10, 9, 10, 9)),
    ]

    for n, expt in col_vals:
        rv = GridStrategy.get_grid(n)
        assert rv == expt, (n, [rv, expt])

    print('All tests passed')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Demonstrate grid strategies')
    parser.add_argument('--test', action='store_true',
                        help='Run tests instead of outputting grids.')
    parser.add_argument('--out-dir', '-o', default='grid_demo',
                        help='Output directory for the grid demo.')

    args = parser.parse_args()

    if args.test:
        run_tests()
    else:
        from matplotlib import gridspec
        from matplotlib import pyplot as plt

        output_demos(args.out_dir)

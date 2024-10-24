from rich import print
from datetime import datetime

rich_print = print

def print(*args, **kw):
    print_time = False

    if 'time' in kw:
        if kw['time'] == True:
            print_time = True

        del kw['time']

    if print_time:
        cur_time = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

        rich_print(f'[not bold][default]{cur_time}[/default][/not bold]',
                   end=' ')

    rich_print(*args, **kw)

def print_split_status(outer_loop: int,
                       split_stats: dict,
                       val_score: float,
                       test_score: float):
    '''
        Print the stats of the current split.
    '''
    print(f"[red][bold]ls outer loop {outer_loop}[/bold][/red] @ "
          f"[not bold][default]"
          f"{datetime.now().strftime('%H:%M:%S %Y/%m/%d')}"
          f"[/not bold][default]")

    print(f"| generalization gap {val_score - test_score:>5.2f} "
          f"(val {val_score:>5.2f}, test {test_score:>5.2f})")
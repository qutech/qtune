import re
import io
import uuid
import datetime

import matlab.engine


def make_view(engine, eval_str):
    """

    :param engine: matlab engine as returned by matlab.engine.connect_matlab()
    :param eval_str: string to evaluate
    :return:
    """
    try:
        with io.StringIO() as dump:
            f_info = engine.eval('functions(@%s)' % eval_str, stdout=dump, stderr=dump)
            if f_info['type'] == 'simple' and f_info['file'] == '':
                pass
            else:
                return MATLABFunctionView(engine, eval_str)
    except matlab.engine.MatlabExecutionError:
        pass
    except SyntaxError:
        pass

    if engine.eval('isstruct(%s)' % eval_str):
        return MATLABStructView(engine, eval_str)
    elif engine.eval('iscell(%s)' % eval_str):
        return MATLABCellView(engine, eval_str)
    elif engine.eval('istable(%s)' % eval_str):
        return MATLABTableView(engine, eval_str)
    else:
        return engine.eval(eval_str)


def size(matlab_view):
    return matlab_view._eval('size(%s)' % matlab_view._eval_str)


def _get_item_array(eval_str, idx):
    return r"subsref(%s, struct('type', {'()'}, 'subs', {'%s'}))" % (eval_str, idx)


def _get_item_cell(eval_str, idx):
    return r"subsref(%s, struct('type', {'{}'}, 'subs', {'%s'}))" % (eval_str, idx)


def _get_field(eval_str, field):
    return r"subsref(%s, struct('type', {'.'}, 'subs', {'%s'}))" % (eval_str, field)


def _set_field(eval_str, field, val_str):
    return r"subsasgn(%s, struct('type', {'.'}, 'subs', {'%s'}), %s)" % (eval_str, field, val_str)


def _transform_int(py_idx):
    if py_idx < 0:
        return 'end - %d' % abs(py_idx)
    else:
        return str(py_idx + 1)


def _get_id():
    unique_part = str(uuid.uuid4().int)

    time_part = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S_%f')[:]

    return time_part + '_' + unique_part[:8]


def _transform_slice(python_slice):
    if python_slice.step is None:
        step = 1
    else:
        step = python_slice.step
    
    if python_slice.start is None:
        start = '1' if step > 0 else 'end'
    elif idx.start < 0:
        start = '(end - %d)' % abs(python_slice.start)
    else:
        start = str(python_slice.start + 1 if step > 0 else python_slice.start)
    
    if python_slice.stop is None:
        stop = 'end' if step > 0 else '1'
    elif python_slice.stop < 0:
        stop = '(end - %d)' % abs(python_slice.stop)
    else:
        stop = str(python_slice.stop if step > 0 else python_slice.stop + 1)
    
    return '%s:%d:%s' % (start, step, stop)


def _transform_tuple(index_tuple):
    transformed = [_transform_int(idx) if isinstance(idx, int) else _transform_slice(idx)
                   for idx in index_tuple]
    return ','.join(transformed)

def transform_index(idx):
    if isinstance(idx, int):
        idx = _transform_int(idx)
    
    elif isinstance(idx, slice):
        idx = _transform_slice(idx)
    
    elif isinstance(idx, tuple):
        idx = _transform_tuple(idx)
    return idx


class MATLABView:
    _html_tag_re = re.compile('<.*?>')

    @classmethod
    def _clean_html(cls, raw_html):
        return re.sub(cls._html_tag_re, '', raw_html)

    def __init__(self, engine, eval_str):
        self._engine = engine
        self._eval_str = eval_str
        self._error_log = []

    def _eval(self, to_eval, stderr=None, stdout=None):
        with io.StringIO() as dump:
            return self._engine.util.py.my_eval(to_eval, stderr=stderr or dump, stdout=stdout or dump)
        
    def __str__(self):
        return self._clean_html(self._engine.evalc('disp(%s)' % self._eval_str))

    def _repr_pretty_(self, p, cycle):
        text = type(self).__name__ + '(\n' + str(self).rstrip() + '\n)'
        p.text(text.strip())
    
    def __len__(self):
        return int(self._eval('numel(%s)' % self._eval_str))


class MATLABStructView(MATLABView):    
    def __getattr__(self, field: str):
        if field.startswith('_'):
            raise AttributeError()
        if field == 'getdoc' and 'getdoc' not in dir(self):
            raise AttributeError()
        if field == 'next' and 'next' not in dir(self):
            raise AttributeError()
        if self._eval('isscalar(%s)' % self._eval_str):
            return make_view(self._engine, _get_field(self._eval_str, field))
        else:
            return (getattr(self[i], field) for i in range(len(self)))
    
    def __getitem__(self, idx):
        idx = transform_index(idx)
        return MATLABStructView(self._engine,
                                _get_item_array(self._eval_str, idx))
    
    def __dir__(self):
        return super().__dir__() + self._eval('fieldnames(%s)' % self._eval_str)


class MATLABCellView(MATLABView):
    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = transform_index(idx)
            return make_view(self._engine, _get_item_cell(self._eval_str, idx))

        elif all(isinstance(dim_idx, int) for dim_idx in idx):
            idx = transform_index(idx)
            return make_view(self._engine, _get_item_cell(self._eval_str, idx))
        
        else:
            raise NotImplementedError()
    
    @property
    def shape(self):
        return self._eval('size(%s)' % self._eval_str)


class MATLABTableView(MATLABView):
    def __getattr__(self, field):
        if field.startswith('_'):
            raise AttributeError()
        if field == 'getdoc' and 'getdoc' not in dir(self):
            raise AttributeError()
        return make_view(self._engine, self._eval_str + '.' + field)

    def __dir__(self):
        return super().__dir__() + self._eval('fieldnames(%s)' % self._eval_str)


class MATLABFunctionView(MATLABView):
    base_path = 'MATLABViewFromPythonData'

    def __call__(self, *args, **kwargs):
        base_path = self.base_path

        if self._engine.exist(base_path, 'var') == 0.:
            self._engine.workspace[base_path] = self._engine.struct()

        empty_call_body = r"struct('args', {{}}, 'result', [])"

        call_id = 'call_' + _get_id() + '_data'
        self._eval("%s = %s;" % (base_path, _set_field(base_path, call_id, empty_call_body)))

        args_path = '.'.join((base_path, call_id, 'args'))
        for arg in args:
            if isinstance(arg, MATLABView):
                temp = arg._eval_str
            else:
                self._engine.workspace['temp'] = arg
                temp = 'temp'

            self._eval(r"%s{end+1} = %s;" % (args_path, temp))

        result_path = '.'.join((base_path, call_id, 'result'))

        self._eval(r"%s = %s(%s{:});" % (result_path, self._eval_str, args_path))

        result = make_view(self._engine, result_path)

        # TODO: when to delete temporary variables?
        # self._eval("%s = rmfield(%s, '%s')" % (base_path, base_path, call_id))
        return result

    def __str__(self):
        return '@' + self._eval_str

    def clear_all_temporary_data(self):
        self._eval("clear('%s')" % self.base_path)


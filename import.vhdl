library ieee;
use ieee.std_logic_1164.all;
use std.textio.all;

entity reader is
end entity;

architecture behaviour of reader is
  -- name of file
  constant file_name: string := "export/weights.txt";
  -- time between batches
  constant batch_interval: time := 1 ns;
  -- size of batches
  constant batch_size: integer := 80;
  -- output signal (batch data)
  signal batch_data: bit_vector(batch_size - 1 downto 0);
begin
  process is
    -- temporary bit vector for reading batch
    variable v_lvec: bit_vector(batch_size - 1 downto 0);
    -- line of text
    variable v_line: line;
    -- file instance
    file v_file: text;
  begin
    -- open file
    file_open(v_file, file_name, read_mode);

    while not endfile(v_file) loop
      -- read line of file
      readline(v_file, v_line);
      -- read line into bit_vector
      hread(v_line, v_lvec);
      -- write bit_vector to output signal
      batch_data <= v_lvec;
      -- wait for interval
      wait for batch_interval;
    end loop;

    -- close file
    file_close(v_file);
    wait;
  end process;
end architecture;